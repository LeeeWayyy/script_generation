"""WhisperX adapter for the versioned Silero weights bundled with faster-whisper."""
from faster_whisper.vad import VadOptions, get_speech_timestamps
from whisperx.diarize import Segment
from whisperx.vads.vad import Vad


class SpeechVad(Vad):
    def __init__(self):
        super().__init__(0.5)

    @staticmethod
    def preprocess_audio(audio):
        return audio

    def __call__(self, audio, **kwargs):
        if audio['sample_rate'] != 16000:
            raise ValueError('Speech detection requires 16-kHz audio')
        spans = get_speech_timestamps(audio['waveform'], VadOptions(max_speech_duration_s=30))
        return [Segment(s['start'] / 16000, s['end'] / 16000, 'UNKNOWN') for s in spans]

    @staticmethod
    def merge_chunks(segments, chunk_size, onset=0.5, offset=None):
        return Vad.merge_chunks(segments, chunk_size, onset, offset) if segments else []
