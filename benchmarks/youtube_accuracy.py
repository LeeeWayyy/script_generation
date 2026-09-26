"""Freeze creator references separately; score audio-only server results."""
import argparse
import hashlib
from importlib.metadata import version
import json
import re
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


def assess(reference, generated, language):
    meta = generated.get('meta', {})
    if meta.get('transcript_source') == 'youtube_manual_captions' or meta.get('caption_language'):
        raise ValueError('Reference leakage: generated output used captions')
    hypothesis = '\n'.join(s['text'] for s in generated['segments'])
    result = score(reference, hypothesis)
    strict_v1 = dict(result)
    normalization = 'benchmarks.run.normalize_text v1; creator annotations retained'
    if language == 'en':
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
        normalizer = EnglishTextNormalizer({})
        expected, actual = normalizer(reference), normalizer(hypothesis)
        reference_words, hypothesis_words = expected.split(), actual.split()
        reference_chars, hypothesis_chars = expected.replace(' ', ''), actual.replace(' ', '')
        if not reference_words:
            raise ValueError('Reference has no words after English normalization')
        word_errors = edit_distance(reference_words, hypothesis_words)
        char_errors = edit_distance(reference_chars, hypothesis_chars)
        result = {'wer': word_errors / len(reference_words),
                  'cer': char_errors / len(reference_chars),
                  'word_errors': word_errors, 'reference_words': len(reference_words),
                  'char_errors': char_errors, 'reference_chars': len(reference_chars),
                  'normalized_hypothesis_sha256': hashlib.sha256(actual.encode()).hexdigest()}
        normalization = ('Whisper EnglishTextNormalizer, empty spelling map; transformers '
                         + version('transformers') + '; bracketed annotations removed')
    metric = 'cer' if language == 'ja' else 'wer'
    return {**result, 'primary_metric': metric, 'strict_v1_score': strict_v1,
            'normalization': normalization,
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
        if path.exists():
            generated = json.loads(path.read_text(encoding='utf-8'))
            row.update(assess(reference, generated, case['language']))
            row['generated_sha256'] = _sha256(path)
        else:
            row['status'] = 'no_generated_result'
        rows.append(row)
    result = {'scoring_version': 2, 'cases': len(cases), 'scored': sum('wer' in r for r in rows),
              'normalized_reference_exact_matches': sum(r.get('normalized_reference_exact_match', False) for r in rows),
              'rows': rows}
    save(results / 'reference-scores-v2.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('freeze', 'score'))
    p.add_argument('manifest', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    result = freeze(a.manifest, a.output) if a.action == 'freeze' else report(a.manifest, a.output)
    print(result)
