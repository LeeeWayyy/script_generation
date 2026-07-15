import subprocess

import pytest

import transcript.ingest as ingest


def test_ytdlp_fetch_returns_completed_download(monkeypatch, tmp_path):
    media = tmp_path / "video.mp4"

    class Process:
        returncode = None

        def __init__(self, *_args, **_kwargs):
            pass

        def communicate(self, timeout=None):
            media.write_bytes(b"ok")
            self.returncode = 0
            return f"{media}\n", ""

    monkeypatch.setenv("TRANSCRIPT_MAX_DOWNLOAD_BYTES", "3")
    monkeypatch.setattr(ingest.subprocess, "Popen", Process)
    assert ingest._ytdlp_fetch(
        "https://example.com/video", fmt="best", out_template=str(tmp_path / "%(id)s"),
        search_dir=tmp_path, fallback_glob="*", fail_action="download",
        nofile_msg="missing",
    ) == media


def test_ytdlp_fetch_stops_and_cleans_unknown_size_download_over_cap(monkeypatch, tmp_path):
    media = tmp_path / "video.mp4"
    captured = {}

    class Process:
        returncode = None
        pid = 123

        def __init__(self, cmd, **kwargs):
            captured.update(cmd=cmd, kwargs=kwargs, process=self)
            self.killed = False
            self.stop_timeout = None

        def communicate(self, timeout=None):
            if self.killed:
                self.returncode = -9
                self.stop_timeout = timeout
                return "", ""
            media.write_bytes(b"xxxx")
            raise subprocess.TimeoutExpired(captured["cmd"], timeout)

        def kill(self):
            self.killed = True

    monkeypatch.setenv("TRANSCRIPT_MAX_DOWNLOAD_BYTES", "3")
    monkeypatch.setenv("TRANSCRIPT_DOWNLOAD_TIMEOUT_S", "7")
    monkeypatch.setattr(ingest.subprocess, "Popen", Process)
    killed_groups = []
    if ingest.os.name == "posix":
        monkeypatch.setattr(ingest.os, "killpg", lambda pid, sig: killed_groups.append((pid, sig)))

    with pytest.raises(RuntimeError, match="3-byte cap"):
        ingest._ytdlp_fetch(
            "https://example.com/video", fmt="best", out_template=str(tmp_path / "%(id)s"),
            search_dir=tmp_path, fallback_glob="*", fail_action="download",
            nofile_msg="missing",
        )

    assert captured["cmd"][captured["cmd"].index("--max-filesize") + 1] == "3"
    assert "--ignore-config" in captured["cmd"]
    assert captured["kwargs"]["text"] is True
    assert captured["process"].stop_timeout == ingest.PROCESS_STOP_TIMEOUT_S
    if ingest.os.name == "posix":
        assert captured["kwargs"]["start_new_session"] is True
        assert killed_groups == [(123, ingest.signal.SIGKILL)]
    assert not media.exists()


def test_ytdlp_runner_caps_metadata_stdout(monkeypatch, tmp_path):
    captured = {}

    class Process:
        returncode = None

        def __init__(self, cmd, **_kwargs):
            captured["cmd"] = cmd
            self.killed = False

        def communicate(self, timeout=None):
            if self.killed:
                self.returncode = -9
                return "", ""
            raise subprocess.TimeoutExpired(captured["cmd"], timeout, output=b"xxxx")

        def kill(self):
            self.killed = True

    monkeypatch.setenv("TRANSCRIPT_MAX_DOWNLOAD_BYTES", "100")
    monkeypatch.setattr(ingest, "MAX_YTDLP_STDOUT_BYTES", 3)
    monkeypatch.setattr(ingest.subprocess, "Popen", Process)

    with pytest.raises(RuntimeError, match="3-byte cap"):
        ingest._run_ytdlp(
            ["python", "-m", "yt_dlp", "--dump-single-json"],
            url="https://youtube.com/watch?v=x", search_dir=tmp_path,
            fallback_glob="*", fail_action="inspect", monitor_stdout=True,
        )

    assert "--max-filesize" not in captured["cmd"]


def test_process_reap_error_does_not_mask_cap_or_skip_cleanup(monkeypatch, tmp_path):
    partial = tmp_path / "video.webm.part"

    class Process:
        returncode = None

        def __init__(self, cmd, **_kwargs):
            self.cmd = cmd
            self.killed = False

        def communicate(self, timeout=None):
            if self.killed:
                raise OSError("pipe cleanup failed")
            partial.write_bytes(b"xxxx")
            raise subprocess.TimeoutExpired(self.cmd, timeout)

        def kill(self):
            self.killed = True

    monkeypatch.setenv("TRANSCRIPT_MAX_DOWNLOAD_BYTES", "3")
    monkeypatch.setattr(ingest.subprocess, "Popen", Process)

    with pytest.raises(RuntimeError, match="3-byte cap"):
        ingest._ytdlp_fetch(
            "https://example.com/video", fmt="best",
            out_template=str(tmp_path / "%(id)s"), search_dir=tmp_path,
            fallback_glob="*", fail_action="download", nofile_msg="missing",
        )

    assert not partial.exists()


def test_ytdlp_timeout_and_nonfinite_config_fail_cleanly(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSCRIPT_DOWNLOAD_TIMEOUT_S", "2")
    partial = tmp_path / "video.webm.part"

    class Process:
        returncode = None

        def __init__(self, *_args, **_kwargs):
            partial.write_bytes(b"partial")

        def communicate(self, timeout=None):
            self.returncode = -9
            return "", ""

        def kill(self):
            pass

    times = iter((0.0, 3.0))
    monkeypatch.setattr(ingest.subprocess, "Popen", Process)
    monkeypatch.setattr(ingest.time, "monotonic", lambda: next(times))
    with pytest.raises(RuntimeError, match="timed out after 2s"):
        ingest._ytdlp_fetch(
            "https://example.com/video", fmt="best", out_template=str(tmp_path / "%(id)s"),
            search_dir=tmp_path, fallback_glob="*", fail_action="download",
            nofile_msg="missing",
        )
    assert not partial.exists()

    monkeypatch.setenv("TRANSCRIPT_DOWNLOAD_TIMEOUT_S", "inf")
    with pytest.raises(RuntimeError, match="greater than zero"):
        ingest._download_limits()


def test_frame_download_has_no_uncapped_format_fallback(monkeypatch, tmp_path):
    captured = {}
    media = tmp_path / "video.mp4"
    media.write_bytes(b"x")

    def fetch(_url, **kwargs):
        captured.update(kwargs)
        return media

    monkeypatch.setattr(ingest, "_ytdlp_fetch", fetch)
    ingest.download_frame_video("https://example.com/video", tmp_path, height_cap=720)
    assert captured["fmt"] == "bestvideo[height<=720]/best[height<=720]"
