import pytest

import transcript
from transcript.engine import TranscriptionEngine
from transcript.types import Transcript


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"device": "rocm"}, "device"),
        ({"batch_size": 0}, "batch_size"),
        ({"min_speakers": 0}, "min_speakers"),
        ({"max_speakers": 0}, "max_speakers"),
        ({"min_speakers": 3, "max_speakers": 2}, "cannot exceed"),
        ({"diarize": False, "min_speakers": 1}, "diarize is false"),
    ),
)
def test_public_options_are_validated_before_source_work(kwargs, message):
    with pytest.raises(ValueError, match=message):
        transcript.transcribe("/does/not/exist", engine=object(), **kwargs)


def test_builtin_engine_requires_hf_token_before_download(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.setattr(
        transcript, "download_manual_caption",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("download started")),
    )

    with pytest.raises(RuntimeError, match="speaker-diarization-community-1"):
        transcript.transcribe(
            "https://youtube.com/watch?v=x", device="cpu", work_dir="/tmp",
        )


def test_custom_engine_can_diarize_without_hf_token(monkeypatch, tmp_path):
    source = tmp_path / "audio.mp3"
    source.write_bytes(b"audio")
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"wav")
    monkeypatch.setattr(transcript, "extract_audio", lambda *_a, **_k: wav)

    class Engine:
        model_name = "custom"
        device = "cpu"
        compute_type = "custom"

        def run(self, *_args, **_kwargs):
            return Transcript()

    assert transcript.transcribe(
        str(source), engine=Engine(), diarize=True, work_dir=str(tmp_path),
    ).meta["diarized"] is True


def test_custom_engine_subclass_can_diarize_without_hf_token(monkeypatch, tmp_path):
    source = tmp_path / "audio.mp3"
    source.write_bytes(b"audio")
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"wav")
    monkeypatch.setattr(transcript, "extract_audio", lambda *_a, **_k: wav)

    class Engine(TranscriptionEngine):
        model_name = "custom"
        device = "cpu"
        compute_type = "custom"
        hf_token = None

        def __init__(self):
            pass

        def run(self, *_args, **_kwargs):
            return Transcript()

    assert transcript.transcribe(
        str(source), engine=Engine(), diarize=True, work_dir=str(tmp_path),
    ).meta["diarized"] is True


def test_mps_remains_a_supported_cpu_fallback(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    engine = TranscriptionEngine(device="mps")
    assert engine.device == "cpu"
