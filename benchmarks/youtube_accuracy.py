"""Freeze creator references separately; score audio-only server results."""
import argparse
import hashlib
from importlib.metadata import version
import json
import re
import unicodedata
import wave
from pathlib import Path

from benchmarks.run import _sha256, edit_distance, normalize_text, score
from benchmarks.youtube_baseline.freeze import verified_audio
from benchmarks.youtube_baseline.run import save


def freeze(manifest, output):
    from transcript.audio import extract_audio
    from transcript.ingest import download_manual_caption, parse_json3_caption

    output.mkdir(parents=True, exist_ok=True)
    pinned_path = output / 'pinned.json'
    data = json.loads(manifest.read_text(encoding='utf-8'))
    pinned = json.loads(pinned_path.read_text(encoding='utf-8')) if pinned_path.exists() else {
        'schema_version': 1, 'source_manifest_sha256': _sha256(manifest), 'cases': [],
    }
    if pinned['source_manifest_sha256'] != _sha256(manifest):
        raise ValueError('Manifest changed; use a new corpus directory')
    existing = {c['id']: c for c in pinned['cases']}
    for case in data['cases']:
        if case['id'] in existing:
            verify_reference(existing[case['id']], output)
            continue
        if not re.fullmatch(r'[a-z0-9-]+', case['id']):
            raise ValueError('Unsafe case identifier')
        folder = output / case['id']
        folder.mkdir(exist_ok=True)
        downloaded = download_manual_caption(case['url'], folder, language=case['language'],
                                             with_audio=True)
        if not downloaded or not downloaded[0] or not downloaded[2]:
            raise ValueError('Creator reference/audio unavailable: ' + case['id'])
        captions, language, media, info = downloaded
        audio_language = info.get('language')
        if audio_language and audio_language.split('-')[0] != case['language']:
            raise ValueError('Reference language differs from audio: ' + case['id'])
        if language not in info.get('subtitles', {}):
            raise ValueError('Reference must be creator-provided')
        reference = '\n'.join(s.text for s in parse_json3_caption(captions))
        if not normalize_text(reference):
            raise ValueError('Empty creator reference')
        reference_path = folder / 'reference.txt'
        reference_path.write_text(reference, encoding='utf-8')
        audio = extract_audio(media, folder)
        with wave.open(str(audio), 'rb') as stream:
            assert (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) == (16000, 1, 2)
            duration = stream.getnframes() / stream.getframerate()
        pinned['cases'].append({
            **case, 'duration_s': duration, 'audio_language': audio_language,
            'media': audio.relative_to(output).as_posix(), 'media_sha256': _sha256(audio),
            'reference': reference_path.relative_to(output).as_posix(),
            'reference_sha256': _sha256(reference_path),
            'reference_captions': captions.relative_to(output).as_posix(),
            'reference_captions_sha256': _sha256(captions), 'reference_language': language,
            'reference_origin': 'creator_provided', 'download_sha256': _sha256(media),
            'reference_independently_reviewed': False,
        })
        save(pinned_path, pinned)
        print('FROZEN', case['id'], round(duration, 2), flush=True)
    return pinned_path


def verify_reference(case, root):
    verified_audio(case, root)
    for field in ('reference', 'reference_captions'):
        path = (root / case[field]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('Reference must remain inside the frozen corpus')
        if _sha256(path) != case[field + '_sha256']:
            raise ValueError('Frozen reference changed: ' + case['id'])
    return (root / case['reference']).read_text(encoding='utf-8')


# Caption annotations are not speech: bracketed sounds/implied words such as
# [LAUGHTER] or （私は）, and line-leading speaker labels such as PROFESSOR:.
# ponytail: space-delimited labels (ja-reference-20 "ursさん ...") are not
# detectable without guessing; they stay in the reference and are documented.
ANNOTATION = re.compile(r'\[[^\]\n]*\]|\([^)\n]*\)|（[^）\n]*）|【[^】\n]*】'
                        r'|^[ \t]*(?:[A-Z][\w.\'-]*(?: [A-Z][\w.\'-]*){0,2}|[^\s\d:：][^\s:：]{0,9})[:：]',
                        re.MULTILINE)


def strip_annotations(text):
    return ANNOTATION.sub(' ', text)


# Fillers/backchannels are not required transcript content. English drops them
# in the Whisper normalizer; Japanese has no word boundaries, so match kana forms.
# ponytail: substring match can hit kana words (うんどう); it is applied to both
# texts, so it only matters when one side writes kana and the other kanji.
JA_FILLER = re.compile(r'え[ー〜~]*っ?と[ー〜~]*|う[ー〜~]*ん|[えあうんま][ー〜~]+|あの[ー〜~]+')


def strip_fillers(text, language):
    return JA_FILLER.sub(' ', unicodedata.normalize('NFKC', text)) if language == 'ja' else text


CURLY_APOSTROPHES = str.maketrans({'\N{RIGHT SINGLE QUOTATION MARK}': "'",
                                   '\N{LEFT SINGLE QUOTATION MARK}': "'"})
ENGLISH_SPELLINGS = {'ok': 'okay'}
CORRECTIONS = Path(__file__).with_name('youtube_accuracy') / 'reference-corrections.json'
KANJI_DIGITS = '〇一二三四五六七八九'


def kanji_number(n):
    """Positional Arabic number as a kanji numeral (3760 -> 三千七百六十)."""
    if n == 0 or n >= 10 ** 16:
        return KANJI_DIGITS[0] if n == 0 else str(n)
    out = ''
    for unit, name in ((10 ** 12, '兆'), (10 ** 8, '億'), (10 ** 4, '万'), (1, '')):
        chunk = n // unit % 10000
        for value, place in ((1000, '千'), (100, '百'), (10, '十'), (1, '')):
            digit = chunk // value % 10
            if digit:
                out += ('' if digit == 1 and place else KANJI_DIGITS[digit]) + place
        out += name if chunk else ''
    return out


def _kanji_numerals(text):
    return re.sub(r'\d[\d,]*', lambda m: kanji_number(int(m.group().replace(',', ''))), text)


def _word_readings(text):
    """(start, end, hiragana) per dictionary word, or None if offsets do not cover text."""
    import pykakasi
    words, start = [], 0
    for item in pykakasi.kakasi().convert(text):
        words.append((start, start + len(item['orig']), item['hira']))
        start += len(item['orig'])
    return words if start == len(text) else None


def _word_span(words, start, end):
    covered = [w for w in words if w[0] < max(end, start + 1) and w[1] > min(start, end - 1)]
    return (min(covered[0][0], start), max(covered[-1][1], end)) if covered else (start, end)


def _spelling_variants(reference, hypothesis, blocks, words):
    """Blocks that only spell the same words differently (時/とき, 綺麗/きれい).

    Each difference is widened by the same equal context on both sides to whole
    dictionary words; differences whose words overlap are judged together. Kanji
    replaced by other kanji stays an error even with the same reading (自転/時点).
    """
    groups, merge_next = [], False
    for index, block in enumerate(blocks):
        i, j, k, m = block[3]
        ref_start, ref_end = _word_span(words[0], i, j)
        hyp_start, hyp_end = _word_span(words[1], k, m)
        need_left, need_right = max(i - ref_start, k - hyp_start), max(ref_end - j, hyp_end - m)
        # Widen only through equal text, whose offsets match on both sides.
        previous = blocks[index - 1][3] if index else (0, 0, 0, 0)
        following = blocks[index + 1][3] if index + 1 < len(blocks) else (len(reference), 0, len(hypothesis), 0)
        gap_left, gap_right = i - previous[1], following[0] - j
        right = min(need_right, gap_right)
        if groups and (merge_next or need_left > gap_left):
            groups[-1][0].append(block)
            groups[-1][1][1], groups[-1][1][3] = j + right, m + right
        else:
            left = min(need_left, gap_left)
            groups.append([[block], [i - left, j + right, k - left, m + right]])
        merge_next = need_right > gap_right
    accepted = []
    for members, (a, b, c, d) in groups:
        if any(KANJI.search(block[1]) and KANJI.search(block[2]) for block in members):
            continue
        if _hiragana(reference[a:b]) == _hiragana(hypothesis[c:d]):
            accepted.extend(members)
    return accepted


def _hiragana(text):
    import pykakasi
    return ''.join(item['hira'] for item in pykakasi.kakasi().convert(text))


KANJI = re.compile(r'[一-鿿]')


def _blocks(reference, hypothesis):
    """Minimal edit script grouped into contiguous differences: (cost, ref, hyp)."""
    from rapidfuzz.distance import Levenshtein
    blocks, current = [], None
    for op in Levenshtein.opcodes(reference, hypothesis):
        if op.tag == 'equal':
            current = None
            continue
        if current is None:
            current = [0, op.src_start, op.src_end, op.dest_start, op.dest_end]
            blocks.append(current)
        current[0] += max(op.src_end - op.src_start, op.dest_end - op.dest_start)
        current[2], current[4] = op.src_end, op.dest_end
    return [(cost, reference[i:j], hypothesis[k:m], (i, j, k, m)) for cost, i, j, k, m in blocks]


def _largest(blocks, joiner, count=8):
    return [{'errors': b[0], 'reference': joiner.join(b[1]), 'generated': joiner.join(b[2])}
            for b in sorted(blocks, key=lambda b: -b[0])[:count]]


def _measure(reference, hypothesis, language):
    if language != 'en':
        expected, actual = (normalize_text(_kanji_numerals(unicodedata.normalize('NFKC', t))).replace(' ', '')
                            for t in (reference, hypothesis))
        if not expected:
            raise ValueError('Reference has no characters after Japanese normalization')
        blocks = _blocks(expected, actual)
        words = (_word_readings(expected), _word_readings(actual))
        # Kanji/kana/katakana spelling of the same word is not a recognition error.
        spelling = _spelling_variants(expected, actual, blocks, words) if None not in words else []
        remaining = [b for b in blocks if b not in spelling]
        errors = sum(b[0] for b in remaining)
        return ({'cer': errors / len(expected), 'wer': errors / len(expected),
                 'char_errors': errors, 'reference_chars': len(expected),
                 'strict_char_errors': errors + sum(b[0] for b in spelling),
                 'acceptable_differences': {'kana_kanji_spelling': sum(b[0] for b in spelling),
                                            'examples': _largest(spelling, '', 5)},
                 'largest_remaining_differences': _largest(remaining, ''),
                 'normalized_hypothesis_sha256': hashlib.sha256(actual.encode()).hexdigest()},
                'benchmarks.run.normalize_text v1; Arabic numbers as kanji numerals; same-reading '
                'kana/kanji spelling tolerated via pykakasi ' + version('pykakasi'))
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    # The normalizer expands only ASCII contractions: it’s would score as "it s".
    normalizer = EnglishTextNormalizer(ENGLISH_SPELLINGS)
    expected, actual = (normalizer(t.translate(CURLY_APOSTROPHES)) for t in (reference, hypothesis))
    reference_words, hypothesis_words = expected.split(), actual.split()
    reference_chars, hypothesis_chars = expected.replace(' ', ''), actual.replace(' ', '')
    if not reference_words:
        raise ValueError('Reference has no words after English normalization')
    blocks = _blocks(reference_words, hypothesis_words)
    # Word spacing alone (vietcong / viet cong) is a spelling choice, not a misheard word.
    spacing = [b for b in blocks if ''.join(b[1]) == ''.join(b[2])]
    remaining = [b for b in blocks if b not in spacing]
    word_errors = sum(b[0] for b in remaining)
    char_errors = edit_distance(reference_chars, hypothesis_chars)
    return ({'wer': word_errors / len(reference_words),
             'cer': char_errors / len(reference_chars),
             'word_errors': word_errors, 'reference_words': len(reference_words),
             'char_errors': char_errors, 'reference_chars': len(reference_chars),
             'strict_word_errors': word_errors + sum(b[0] for b in spacing),
             'acceptable_differences': {'word_spacing': sum(b[0] for b in spacing),
                                        'examples': _largest(spacing, ' ', 5)},
             'largest_remaining_differences': _largest(remaining, ' '),
             'normalized_hypothesis_sha256': hashlib.sha256(actual.encode()).hexdigest()},
            'Whisper EnglishTextNormalizer, spelling map ok=okay, ASCII apostrophes; transformers '
            + version('transformers') + '; bracketed annotations removed; word spacing tolerated')


def correct_reference(case_id, reference):
    """Apply recorded, reviewable reference corrections; frozen files stay unchanged."""
    applied = []
    corrections = json.loads(CORRECTIONS.read_text(encoding='utf-8')) if CORRECTIONS.exists() else {}
    for correction in corrections.get(case_id, []):
        reference, removed = re.subn(correction['pattern'], ' ', reference, flags=re.MULTILINE)
        applied.append({**correction, 'removed': removed})
    return reference, applied


def assess(reference, generated, language, span=None):
    meta = generated.get('meta', {})
    if meta.get('transcript_source') == 'youtube_manual_captions' or meta.get('caption_language'):
        raise ValueError('Reference leakage: generated output used captions')
    segments = generated['segments']
    outside = []
    if span:
        # Captions cannot judge generated text before/after the span they cover.
        low, high = span[0] - 1, span[1] + 1
        outside = [s for s in segments if s.get('end', low) < low or s.get('start', high) > high]
        segments = [s for s in segments if s not in outside]
    hypothesis = '\n'.join(s['text'] for s in segments)
    annotated, _ = _measure(reference, hypothesis, language)
    cleaned_reference = strip_fillers(strip_annotations(reference), language)
    cleaned_hypothesis = strip_fillers(strip_annotations(hypothesis), language)
    result, normalization = _measure(cleaned_reference, cleaned_hypothesis, language)
    metric = 'cer' if language == 'ja' else 'wer'
    for field in ('largest_remaining_differences', 'acceptable_differences'):
        annotated.pop(field, None)
    return {**result, 'primary_metric': metric, 'strict_v1_score': score(reference, hypothesis),
            'annotations_retained_v2_score': annotated,
            'generated_chars_outside_reference_span': sum(len(s['text']) for s in outside),
            'normalization': normalization + '; caption annotations, speaker labels and fillers removed (v4)',
            'normalized_reference_exact_match': result[metric] == 0,
            'reference_status': 'creator_provided_not_independently_reviewed',
            'speaker_accuracy': None, 'acoustic_timing_accuracy': None,
            'spoken_only_accuracy': None}


def report(manifest, results):
    cases = json.loads(manifest.read_text(encoding='utf-8'))['cases']
    rows = []
    for case in cases:
        reference = verify_reference(case, manifest.parent)
        path = results / (case['id'] + '.raw.json')
        row = {'case': case['id'], 'language': case['language'], 'split': case['split']}
        if case.get('reference_quality_note'):
            row['reference_quality_note'] = case['reference_quality_note']
        if 'duration_s' in case:
            from transcript.ingest import parse_json3_caption
            captions = parse_json3_caption(manifest.parent / case['reference_captions'])
            first = min((s.start for s in captions), default=0)
            last = max((s.end for s in captions), default=0)
            row['reference_caption_bounds_s'] = [first, last]
            # A screening warning, not a claim that every gap contains speech.
            if first > case['duration_s'] * .2 or last < case['duration_s'] * .8:
                row['partial_reference_coverage'] = True
                row['reference_quality_note'] = (row.get('reference_quality_note', '')
                    + ' Caption timestamps cover only part of the media; review before using as full-video ground truth.').strip()
        reference, corrections = correct_reference(case['id'], reference)
        if corrections:
            row['reference_corrections'] = corrections
        if path.exists():
            generated = json.loads(path.read_text(encoding='utf-8'))
            row.update(assess(reference, generated, case['language'],
                              row.get('reference_caption_bounds_s')))
            row['generated_sha256'] = _sha256(path)
        else:
            row['status'] = 'no_generated_result'
        rows.append(row)
    result = {'scoring_version': 4, 'cases': len(cases), 'scored': sum('wer' in r for r in rows),
              'normalized_reference_exact_matches': sum(r.get('normalized_reference_exact_match', False) for r in rows),
              'rows': rows}
    result['groups'] = []
    for language, split in sorted({(r['language'], r['split']) for r in rows}):
        group = [r for r in rows if (r['language'], r['split']) == (language, split)]
        scored = [r for r in group if 'wer' in r]
        unit = 'char' if language == 'ja' else 'word'
        denominator = sum(r['reference_' + unit + 's'] for r in scored)
        full = [r for r in scored if not r.get('partial_reference_coverage')]
        full_denominator = sum(r['reference_' + unit + 's'] for r in full)
        result['groups'].append({
            'language': language, 'split': split, 'cases': len(group), 'scored': len(scored),
            'primary_metric': 'cer' if language == 'ja' else 'wer',
            'reference_weighted_error_rate': (sum(r[unit + '_errors'] for r in scored) / denominator
                                               if denominator else None),
            'reference_quality_flagged': sum('reference_quality_note' in r for r in group),
            # A partial reference scores every later generated word as an insertion.
            'full_coverage_reference_weighted_error_rate': (
                sum(r[unit + '_errors'] for r in full) / full_denominator if full_denominator else None),
            'partial_coverage_excluded': len(scored) - len(full),
        })
    save(results / 'reference-scores-v4.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('freeze', 'score'))
    p.add_argument('manifest', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    result = freeze(a.manifest, a.output) if a.action == 'freeze' else report(a.manifest, a.output)
    print(result)
