from copy import deepcopy
from benchmarks.youtube_baseline.run import assess


def test_baseline_rejects_contract_regressions_without_claiming_accuracy():
    raw = {'language': 'en', 'segments': [{'text': 'Hello.', 'start': 0., 'end': 1.,
            'speaker': 'A', 'words': [{'word': 'Hello.', 'start': 0., 'end': 1., 'speaker': 'A'}]}],
           'meta': {'align_succeeded': True, 'diarize_succeeded': True}}
    readable = deepcopy(raw)
    readable['meta']['readable'] = {'fallbacks': []}
    case = {'language': 'en', 'duration_s': 1.}
    result = assess(raw, readable, case)
    assert result['functional_pass'] and result['wer'] is None and result['cer'] is None
    readable['meta']['readable']['fallbacks'] = [{'segment_index': 0}] * 2
    assert 'duplicate_fallback_index' in assess(raw, readable, case)['errors']
    readable['segments'][0]['words'][0]['end'] = -1
    result = assess(raw, readable, case)
    assert not result['functional_pass']
    assert 'invalid_or_out_of_media_timing' in result['errors']
    assert 'readable_changed_word_data' in result['errors']
    raw['meta']['transcript_source'] = 'youtube_manual_captions'
    assert 'creator_captions_used' in assess(raw, readable, case)['errors']


def test_frozen_corpus_has_twenty_per_language_and_caption_evidence():
    import json
    from collections import Counter
    from pathlib import Path
    manifest = json.loads((Path(__file__).parents[1] / 'benchmarks/youtube_baseline/manifest.json').read_text(encoding='utf-8'))
    cases = manifest['cases']
    assert Counter(c['language'] for c in cases) == {'en': 20, 'ja': 20}
    assert len({c['video_id'] for c in cases}) == 40
    for language in ('en', 'ja'):
        group = [c for c in cases if c['language'] == language]
        assert len({c['topic'] for c in group}) == 10
        assert len({c['format'] for c in group}) >= 6
        assert {c['length_bucket'] for c in group} == {'short', 'medium', 'long', 'extended'}
        assert Counter(c['split'] for c in group) == {'calibration': 14, 'holdout': 6}
        assert all(c['creator_caption_languages'] == [] and c['caption_checked_at'] for c in group)


def test_pinned_input_rejects_changed_audio(tmp_path):
    import hashlib
    import pytest
    from benchmarks.youtube_baseline.freeze import verified_audio
    audio = tmp_path / 'sample.wav'
    audio.write_bytes(b'fixed audio')
    case = {'id': 'sample', 'media': audio.name,
            'media_sha256': hashlib.sha256(audio.read_bytes()).hexdigest()}
    assert verified_audio(case, tmp_path) == audio
    audio.write_bytes(b'changed audio')
    with pytest.raises(ValueError, match='Pinned audio changed'):
        verified_audio(case, tmp_path)


def test_repeat_comparison_ignores_execution_identity_but_not_caption_changes():
    from benchmarks.youtube_baseline.repeat import semantic_result
    a = {'segments': [{'text': 'Hello.', 'start': 0, 'end': 1}], 'language': 'en',
         'meta': {'job_id': 'first', 'source': 'upload1.wav', 'model': 'large-v3'}}
    b = deepcopy(a)
    b['meta'].update(job_id='second', source='upload2.wav')
    assert semantic_result(a) == semantic_result(b)
    b['segments'][0]['end'] = 1.001
    assert semantic_result(a) != semantic_result(b)
