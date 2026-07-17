import subprocess
from pathlib import Path

import pytest

import transcript.audio as audio


def test_audio_extraction_uses_bounded_process_group(monkeypatch, tmp_path):
    media = tmp_path / "input.mp4"
    captured = {}

    class Process:
        returncode = None

        def __init__(self, cmd, **kwargs):
            captured.update(cmd=cmd, kwargs=kwargs)
            self.output = Path(cmd[-1])

        def communicate(self, timeout=None):
            self.output.write_bytes(b"wav")
            self.returncode = 0
            return None, ""

    monkeypatch.setattr(audio, "ensure_tool", lambda _name: None)
    monkeypatch.setenv("TRANSCRIPT_MAX_AUDIO_BYTES", "4")
    monkeypatch.setattr(audio.subprocess, "Popen", Process)

    result = audio.extract_audio(media, tmp_path)

    assert result.read_bytes() == b"wav"
    assert captured["kwargs"]["stderr"] is subprocess.PIPE
    assert captured["kwargs"]["text"] is True
    assert captured["cmd"][captured["cmd"].index("-fs") + 1] == "5"
    if audio.os.name == "posix":
        assert captured["kwargs"]["start_new_session"] is True


def test_audio_cap_stops_process_and_cleans_partial_output(monkeypatch, tmp_path):
    media = tmp_path / "input.mp4"
    output = tmp_path / "input.16k.wav"
    captured = {}

    class Process:
        returncode = None
        pid = None
        stdout = None
        stderr = None

        def __init__(self, cmd, **_kwargs):
            self.cmd = cmd
            self.output = Path(cmd[-1])
            self.killed = False
            captured["process"] = self

        def communicate(self, timeout=None):
            if self.killed:
                self.returncode = -9
                return None, ""
            self.output.write_bytes(b"xxxx")
            raise subprocess.TimeoutExpired(self.cmd, timeout)

        def kill(self):
            self.killed = True

    monkeypatch.setattr(audio, "ensure_tool", lambda _name: None)
    monkeypatch.setenv("TRANSCRIPT_MAX_AUDIO_BYTES", "3")
    monkeypatch.setattr(audio.subprocess, "Popen", Process)

    with pytest.raises(RuntimeError, match="3-byte cap"):
        audio.extract_audio(media, tmp_path)

    assert captured["process"].killed is True
    assert not output.exists()


def test_audio_failure_preserves_existing_output(monkeypatch, tmp_path):
    media = tmp_path / "input.mp4"
    output = tmp_path / "input.16k.wav"
    output.write_bytes(b"existing")
    captured = {}

    class Process:
        returncode = None

        def __init__(self, cmd, **_kwargs):
            captured["temp"] = Path(cmd[-1])

        def communicate(self, timeout=None):
            self.returncode = 1
            return None, "invalid input"

    monkeypatch.setattr(audio, "ensure_tool", lambda _name: None)
    monkeypatch.setattr(audio.subprocess, "Popen", Process)

    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        audio.extract_audio(media, tmp_path)

    assert output.read_bytes() == b"existing"
    assert not captured["temp"].exists()


def test_audio_timeout_and_invalid_limits_fail_cleanly(monkeypatch, tmp_path):
    media = tmp_path / "input.mp4"
    output = tmp_path / "input.16k.wav"

    class Process:
        returncode = None
        pid = None
        stdout = None
        stderr = None

        def __init__(self, *_args, **_kwargs):
            self.killed = False

        def communicate(self, timeout=None):
            if self.killed:
                self.returncode = -9
                return None, ""
            raise subprocess.TimeoutExpired("ffmpeg", timeout)

        def kill(self):
            self.killed = True

    monkeypatch.setattr(audio, "ensure_tool", lambda _name: None)
    monkeypatch.setenv("TRANSCRIPT_AUDIO_TIMEOUT_S", "2")
    monkeypatch.setattr(audio.subprocess, "Popen", Process)
    times = iter((0.0, 3.0))
    monkeypatch.setattr(audio.time, "monotonic", lambda: next(times))

    with pytest.raises(RuntimeError, match="timed out after 2s"):
        audio.extract_audio(media, tmp_path)
    assert not output.exists()

    monkeypatch.setenv("TRANSCRIPT_AUDIO_TIMEOUT_S", "inf")
    with pytest.raises(RuntimeError, match="greater than zero"):
        audio._audio_limits()
