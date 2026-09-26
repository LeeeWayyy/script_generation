"""Run with the server ML dependencies; client-only installations skip this check."""
import pytest

pytest.importorskip('whisperx')
np = pytest.importorskip('numpy')


def test_speech_detector_rejects_silence_and_wrong_sample_rate():
    from transcript.speech import SpeechVad

    detector = SpeechVad()
    silence = np.zeros(16000, dtype=np.float32)
    assert detector({'waveform': silence, 'sample_rate': 16000}) == []
    assert detector.merge_chunks([], 30) == []
    assert detector.preprocess_audio(silence) is silence
    with pytest.raises(ValueError, match='16-kHz'):
        detector({'waveform': silence, 'sample_rate': 8000})
