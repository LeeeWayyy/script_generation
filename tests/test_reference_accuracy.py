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
