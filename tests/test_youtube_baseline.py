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
