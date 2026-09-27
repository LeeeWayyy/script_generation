import pytest

from benchmarks.run import edit_distance
from benchmarks.youtube_accuracy import assess, verify_reference


def test_reference_scoring_detects_errors_and_caption_leakage():
    generated = {'segments': [{'text': '地球の時点'}], 'meta': {}}
    result = assess('地球の自転', generated, 'ja')
    assert result['char_errors'] == 2 and result['primary_metric'] == 'cer'
    assert not result['normalized_reference_exact_match']
    assert result['speaker_accuracy'] is None
    generated['meta']['transcript_source'] = 'youtube_manual_captions'
    with pytest.raises(ValueError, match='leakage'):
        assess('地球の自転', generated, 'ja')
    assert edit_distance(['a', 'b'], ['a', 'c', 'b']) == 1


def test_reference_hashes_cannot_change_silently(tmp_path):
    from benchmarks.run import _sha256
    case = {'id': 'test'}
    for field in ('media', 'reference', 'reference_captions'):
        path = tmp_path / field
        path.write_text('fixed', encoding='utf-8')
        case[field] = field
        case[field + '_sha256'] = _sha256(path)
    assert verify_reference(case, tmp_path) == 'fixed'
    (tmp_path / 'reference').write_text('corrected without recording', encoding='utf-8')
    with pytest.raises(ValueError, match='reference changed'):
        verify_reference(case, tmp_path)


def test_english_normalization_preserves_currency_and_numeric_errors():
    pytest.importorskip('transformers.models.whisper.english_normalizer')
    def measure(reference, text):
        return assess(reference, {'segments': [{'text': text}]}, 'en')
    correct = measure('ten thousand dollars and 100 percent', '$10,000 and 100%')
    assert correct['wer'] == 0 and correct['strict_v1_score']['wer'] > 0
    assert measure('sixteen', 'sixty')['wer'] > 0
    assert measure('ten thousand dollars', '10,000')['wer'] > 0
    assert measure('It’s OK, don’t worry', "It's okay, don't worry")['wer'] == 0


def test_report_keeps_missing_cases_and_reference_quality_flags(tmp_path):
    import json
    from benchmarks.run import _sha256
    from benchmarks.youtube_accuracy import report
    cases = []
    for name in ('done', 'missing'):
        case = {'id': name, 'language': 'ja', 'split': 'calibration',
                'duration_s': 100}
        for field in ('media', 'reference', 'reference_captions'):
            path = tmp_path / (name + field)
            text = (json.dumps({'events': [{'tStartMs': 0, 'dDurationMs': 20000,
                                           'segs': [{'utf8': '地球の自転'}]}]})
                    if field == 'reference_captions' else '地球の自転')
            path.write_text(text, encoding='utf-8')
            case[field], case[field + '_sha256'] = path.name, _sha256(path)
        cases.append(case)
    manifest = tmp_path / 'pinned.json'
    manifest.write_text(json.dumps({'cases': cases}))
    (tmp_path / 'done.raw.json').write_text(json.dumps({'segments': [{'text': '地球の時点'}]}))
    result = report(manifest, tmp_path)
    assert result['scored'] == 1 and result['cases'] == 2
    assert result['groups'][0]['reference_weighted_error_rate'] == 2 / 5
    assert result['groups'][0]['reference_quality_flagged'] == 2
    assert result['rows'][0]['reference_caption_bounds_s'] == [0, 20]
    assert result['groups'][0]['full_coverage_reference_weighted_error_rate'] is None
    assert result['groups'][0]['partial_coverage_excluded'] == 1
    assert result['rows'][1]['status'] == 'no_generated_result'


def test_caption_annotations_and_speaker_labels_are_not_scored_as_speech():
    from benchmarks.youtube_accuracy import strip_annotations
    generated = {'segments': [{'text': '衣替えしたから'}, {'text': 'しますね'}]}
    result = assess('（私は）衣替えしたから\n（それぐらいの頻度で）しますね', generated, 'ja')
    assert result['cer'] == 0 and result['annotations_retained_v2_score']['cer'] > 0
    assert strip_annotations('PROFESSOR: Good afternoon.\n[LAUGHTER]').split() == ['Good', 'afternoon.']
    # Times and ordinary prose are not labels.
    assert strip_annotations('10:30 we met') == '10:30 we met'
    assert assess('地球の自転', {'segments': [{'text': '地球の時点'}]}, 'ja')['char_errors'] == 2


def test_japanese_fillers_and_backchannels_are_not_required():
    generated = {'segments': [{'text': 'そうですね、地球の自転です。'}]}
    result = assess('うんうんえっとそうですねあのー地球のえー自転ですうーん', generated, 'ja')
    assert result['cer'] == 0 and result['annotations_retained_v2_score']['cer'] > 0
    # Real words are still scored.
    assert assess('あの人', {'segments': [{'text': '人'}]}, 'ja')['char_errors'] == 2


def test_japanese_spelling_variants_are_accepted_but_homophones_are_errors():
    pytest.importorskip('pykakasi')
    def errors(reference, text):
        return assess(reference, {'segments': [{'text': text}]}, 'ja')
    assert errors('3,760円です', '三千七百六十円です')['char_errors'] == 0
    for reference, text in (('小さい時に行った', '小さいときにいった'), ('綺麗な花', 'きれいな花'),
                            ('タメになる', 'ためになる')):
        result = errors(reference, text)
        assert result['char_errors'] == 0 and result['acceptable_differences']['kana_kanji_spelling'] > 0
    # Different kanji with the same reading is a recognition error, not spelling.
    assert errors('台風が来る', '大風が来る')['char_errors'] == 1
    assert errors('そうですねなるほど', 'そうですね')['largest_remaining_differences'][0]['reference'] == 'なるほど'


def test_english_word_spacing_is_accepted_and_listed():
    pytest.importorskip('transformers.models.whisper.english_normalizer')
    result = assess('the vietcong came', {'segments': [{'text': 'The Viet Cong came'}]}, 'en')
    assert result['wer'] == 0 and result['strict_word_errors'] == 2
    assert result['acceptable_differences']['word_spacing'] == 2


def test_reference_corrections_and_caption_span_are_recorded(tmp_path, monkeypatch):
    import json
    import benchmarks.youtube_accuracy as accuracy
    corrections = tmp_path / 'corrections.json'
    corrections.write_text(json.dumps({'case': [{'pattern': '^noriko(?= |$)', 'reason': 'label'}]}),
                           encoding='utf-8')
    monkeypatch.setattr(accuracy, 'CORRECTIONS', corrections)
    text, applied = accuracy.correct_reference('case', 'noriko こんにちは\nnoriko先生')
    assert text.split() == ['こんにちは', 'noriko先生'] and applied[0]['removed'] == 1
    generated = {'segments': [{'text': '地球の自転', 'start': 1, 'end': 2},
                              {'text': '音楽', 'start': 50, 'end': 60}]}
    result = assess('地球の自転', generated, 'ja', [0, 20])
    assert result['cer'] == 0 and result['generated_chars_outside_reference_span'] == 2
