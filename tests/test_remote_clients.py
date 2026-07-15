"""Remote clients fail locally/cleanly and apply one timeout policy."""

from types import SimpleNamespace

import pytest

from transcript import extract_remote, remote
from transcript import _remote_http
from transcript._remote_http import get_with_retry, poll_until_done, submit_job


class _Response:
    status_code = 200
    ok = True
    text = "result\n"

    def __init__(self, payload=None):
        self.payload = payload

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def test_transcript_client_uses_timeout_for_every_request(monkeypatch, capsys):
    calls = []

    class Requests:
        RequestException = OSError
        ConnectionError = OSError
        Timeout = TimeoutError

        @staticmethod
        def post(url, **kwargs):
            calls.append((url, kwargs["timeout"]))
            return _Response({"id": "0123456789ab"})

        @staticmethod
        def get(url, **kwargs):
            calls.append((url, kwargs["timeout"]))
            if url.endswith("/result"):
                return _Response()
            return _Response({"status": "done"})

    monkeypatch.setitem(__import__("sys").modules, "requests", Requests)
    assert remote.main(["https://example.com/media.mp4", "--timeout", "9", "-q"]) == 0
    assert len(calls) == 3
    assert all(timeout[0] <= 9 and timeout[1] <= 9 for _, timeout in calls)
    assert capsys.readouterr().out == "result\n"


@pytest.mark.parametrize("value", ["0", "nan", "inf"])
def test_transcript_client_rejects_invalid_timeout_locally(value, capsys):
    assert remote.main(["https://example.com/media.mp4", "--timeout", value]) == 1
    assert "--timeout" in capsys.readouterr().err


def test_transcript_client_handles_malformed_submit_json(monkeypatch, capsys):
    class Requests:
        RequestException = OSError
        ConnectionError = OSError
        Timeout = TimeoutError

        @staticmethod
        def post(*args, **kwargs):
            return _Response(ValueError("not json"))

    monkeypatch.setitem(__import__("sys").modules, "requests", Requests)
    assert remote.main(["https://example.com/media.mp4", "-q"]) == 1
    assert "malformed JSON" in capsys.readouterr().err


def test_poll_surfaces_structured_error_reason():
    class Requests:
        RequestException = OSError

        @staticmethod
        def get(*args, **kwargs):
            return _Response({
                "status": "error", "error": "multiple episodes matched",
                "error_reason": "ambiguous",
            })

    with pytest.raises(RuntimeError, match=r"\[ambiguous\].*multiple episodes"):
        poll_until_done(Requests, "https://example/status", {}, poll=1, timeout=5,
                        note=lambda _: None)


def test_poll_rejects_unknown_status():
    class Requests:
        RequestException = OSError

        @staticmethod
        def get(*args, **kwargs):
            return _Response({"status": "paused"})

    with pytest.raises(RuntimeError, match="invalid status 'paused'"):
        poll_until_done(Requests, "https://example/status", {}, poll=1, timeout=5,
                        note=lambda _: None)


def test_poll_retries_only_transient_get_failures_and_reports_stages(monkeypatch):
    calls = []
    closed = []
    responses = [
        OSError("proxy reset"),
        SimpleNamespace(
            status_code=503, ok=False, text="busy", close=lambda: closed.append(True),
        ),
        _Response({"status": "running", "stage": "downloading"}),
        _Response({"status": "running", "stage": "transcribing"}),
        _Response({"status": "done", "stage": "persisting"}),
    ]

    class Requests:
        RequestException = OSError
        ConnectionError = OSError
        Timeout = TimeoutError

        @staticmethod
        def get(*args, **kwargs):
            calls.append(1)
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    monkeypatch.setattr(_remote_http.time, "sleep", lambda _: None)
    notes = []
    assert poll_until_done(
        Requests, "https://example/status", {}, poll=0.01, timeout=5,
        note=notes.append,
    )["status"] == "done"
    assert len(calls) == 5
    assert closed == [True]
    assert notes == [
        "  status: running", "  stage: downloading", "  stage: transcribing",
        "  status: done", "  stage: persisting",
    ]


def test_poll_does_not_retry_nontransient_http_error(monkeypatch):
    calls = []

    class Requests:
        RequestException = OSError

        @staticmethod
        def get(*args, **kwargs):
            calls.append(1)
            return SimpleNamespace(status_code=500, ok=False, text="broken")

    monkeypatch.setattr(_remote_http.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="server returned 500"):
        poll_until_done(Requests, "https://example/status", {}, poll=1, timeout=5,
                        note=lambda _: None)
    assert len(calls) == 1


def test_get_retry_stops_at_the_existing_deadline(monkeypatch):
    now = [0.0]
    calls = []

    class Requests:
        RequestException = OSError

        @staticmethod
        def get(*args, **kwargs):
            calls.append(1)
            return SimpleNamespace(status_code=503)

    monkeypatch.setattr(_remote_http.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(_remote_http.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    with pytest.raises(RuntimeError, match="timed out.*503"):
        get_with_retry(Requests, "https://example", deadline=2.0)
    assert len(calls) == 2


def test_get_does_not_retry_permanent_request_error(monkeypatch):
    class PermanentRequestError(Exception):
        pass

    calls = []

    class Requests:
        RequestException = PermanentRequestError
        ConnectionError = ConnectionError
        Timeout = TimeoutError

        @staticmethod
        def get(*args, **kwargs):
            calls.append(1)
            raise PermanentRequestError("invalid URL")

    monkeypatch.setattr(_remote_http.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="failed: invalid URL"):
        get_with_retry(
            Requests, "bad://url", deadline=_remote_http.time.monotonic() + 100.0,
        )
    assert len(calls) == 1


def test_interrupted_chunked_get_is_retryable():
    import requests

    from transcript._remote_http import is_transient_get_error

    error = requests.exceptions.ChunkedEncodingError("stream ended early")
    assert is_transient_get_error(requests, error)


def test_certificate_error_is_not_retryable():
    import requests

    from transcript._remote_http import is_transient_get_error

    error = requests.exceptions.SSLError("certificate verify failed")
    assert not is_transient_get_error(requests, error)


def test_submit_is_never_retried():
    calls = []

    class Requests:
        RequestException = OSError
        ConnectionError = OSError
        Timeout = TimeoutError

        @staticmethod
        def post(*args, **kwargs):
            calls.append(1)
            raise OSError("connection reset")

    with pytest.raises(RuntimeError, match="could not reach server"):
        submit_job(Requests, "https://example/jobs", data={}, files=None,
                   headers={}, timeout=5)
    assert len(calls) == 1


def test_extraction_client_sends_speaker_and_music_flags(monkeypatch):
    captured = {}

    class Requests:
        RequestException = OSError
        ConnectionError = OSError
        Timeout = TimeoutError

        @staticmethod
        def post(url, **kwargs):
            captured.update(kwargs)
            return _Response({"id": "0123456789ab"})

        @staticmethod
        def get(*args, **kwargs):
            raise SystemExit(0)

    monkeypatch.setitem(__import__("sys").modules, "requests", Requests)
    monkeypatch.setattr(extract_remote, "poll_until_done", lambda *a, **k: {"status": "done"})
    with pytest.raises(SystemExit):
        extract_remote.main([
            "--kind", "audio_extraction", "--feed-url", "https://example.com/feed",
            "--episode-guid", "ep-1", "--episode-url", "https://example.com/episode",
            "--min-speakers", "2", "--max-speakers", "4",
            "--detect-music", "--no-align",
            "--timeout", "12",
        ])
    assert captured["data"]["min_speakers"] == "2"
    assert captured["data"]["max_speakers"] == "4"
    assert captured["data"]["detect_music"] == "true"
    assert captured["data"]["align"] == "false"
    assert captured["data"]["episode_url"] == "https://example.com/episode"
    assert captured["timeout"] == (12.0, 12.0)


def test_extraction_client_rejects_conflicting_options(capsys):
    assert extract_remote.main([
        "--kind", "video", "https://example.com/video.mp4", "--cadence", "2",
    ]) == 1
    assert "--cadence requires --frames" in capsys.readouterr().err

    assert extract_remote.main([
        "--kind", "video", "https://example.com/video.mp4", "--frames", "--cadence", "0.1",
    ]) == 1
    assert "at least 0.5" in capsys.readouterr().err


def test_extraction_client_requires_complete_feed_selector(capsys):
    assert extract_remote.main([
        "--kind", "audio_extraction", "--feed-url", "https://example.com/feed",
        "--episode-title", "Episode",
    ]) == 1
    assert "title and --episode-published" in capsys.readouterr().err


def test_extraction_client_rejects_fallback_with_stronger_selector(capsys):
    assert extract_remote.main([
        "--kind", "audio_extraction", "--feed-url", "https://example.com/feed",
        "--episode-guid", "guid", "--episode-title", "Episode",
        "--episode-published", "2026-01-01",
    ]) == 1
    assert "fallback cannot be combined" in capsys.readouterr().err


def test_unpack_rejects_non_zip_as_verification_error(tmp_path):
    with pytest.raises(extract_remote.BundleVerificationError, match="invalid zip"):
        extract_remote.unpack_and_verify(b"not a zip", tmp_path / "out")


def test_extraction_client_caps_streamed_bundle(monkeypatch, tmp_path, capsys):
    class BundleResponse:
        status_code = 200
        ok = True
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @staticmethod
        def iter_content(chunk_size):
            yield b"1234"

    class Requests:
        RequestException = OSError

        @staticmethod
        def post(*args, **kwargs):
            return _Response({"id": "0123456789ab"})

        @staticmethod
        def get(*args, **kwargs):
            return BundleResponse()

    monkeypatch.setitem(__import__("sys").modules, "requests", Requests)
    monkeypatch.setattr(extract_remote, "poll_until_done", lambda *a, **k: {"status": "done"})
    monkeypatch.setattr(extract_remote, "_MAX_BUNDLE_BYTES", 3)
    assert extract_remote.main([
        "https://example.com/video.mp4", "--kind", "video", "--out-dir", str(tmp_path),
        "-q",
    ]) == 1
    assert "safety limit" in capsys.readouterr().err


def test_transcript_resume_skips_post_and_keeps_job_id_in_quiet_failure(
    monkeypatch, capsys,
):
    job_id = "0123456789ab"

    class Requests:
        RequestException = OSError

        @staticmethod
        def post(*args, **kwargs):
            raise AssertionError("resume must not submit")

    monkeypatch.setitem(__import__("sys").modules, "requests", Requests)
    monkeypatch.setattr(
        remote, "poll_until_done", lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("timed out")
        ),
    )
    assert remote.main(["--job-id", job_id, "--quiet"]) == 1
    assert job_id in capsys.readouterr().err


@pytest.mark.parametrize("client", [remote, extract_remote])
def test_resume_rejects_invalid_or_submission_options(client, capsys):
    assert client.main(["--job-id", "../bad"]) == 1
    assert "invalid job id" in capsys.readouterr().err
    assert client.main(["source.mp4", "--job-id", "0123456789ab"]) == 1
    assert "cannot be combined" in capsys.readouterr().err


def test_transcript_no_align_is_submitted(monkeypatch, capsys):
    captured = {}

    class Requests:
        RequestException = OSError

        @staticmethod
        def post(*args, **kwargs):
            captured.update(kwargs)
            return _Response({"id": "0123456789ab"})

        @staticmethod
        def get(url, **kwargs):
            if url.endswith("/result"):
                return _Response()
            return _Response({"status": "done"})

    monkeypatch.setitem(__import__("sys").modules, "requests", Requests)
    assert remote.main(["https://example/video.mp4", "--no-align", "-q"]) == 0
    assert captured["data"]["align"] == "false"
    assert capsys.readouterr().out == "result\n"


def test_transcript_output_oserror_is_clean_and_names_job(monkeypatch, tmp_path, capsys):
    job_id = "0123456789ab"

    class Requests:
        RequestException = OSError

        @staticmethod
        def get(url, **kwargs):
            if url.endswith("/result"):
                return _Response()
            return _Response({"status": "done"})

    monkeypatch.setitem(__import__("sys").modules, "requests", Requests)
    assert remote.main([
        "--job-id", job_id, "--output", str(tmp_path / "missing" / "out.txt"),
        "--quiet",
    ]) == 1
    error = capsys.readouterr().err
    assert "could not write output" in error
    assert job_id in error


def test_extraction_resume_retries_bundle_and_warns_even_when_quiet(
    monkeypatch, tmp_path, capsys,
):
    job_id = "0123456789ab"
    calls = []

    class BundleResponse:
        status_code = 200
        ok = True
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @staticmethod
        def iter_content(chunk_size):
            yield b"bundle"

    class Requests:
        RequestException = OSError
        ConnectionError = OSError
        Timeout = TimeoutError

        @staticmethod
        def post(*args, **kwargs):
            raise AssertionError("resume must not submit")

        @staticmethod
        def get(url, **kwargs):
            calls.append(url)
            if url.endswith(job_id):
                return _Response({"status": "done"})
            bundle_attempt = sum(call.endswith("/bundle") for call in calls)
            if bundle_attempt == 1:
                raise OSError("proxy reset")
            if bundle_attempt == 2:
                return SimpleNamespace(status_code=503, ok=False, text="busy")
            return BundleResponse()

    envelope = {
        "text": "result",
        "assets": [],
        "meta": {
            "ocr_warning": "OCR unavailable",
            "align_requested": True,
            "align_succeeded": False,
            "music_detection_requested": True,
            "music_detection_succeeded": False,
            "frame_cap_reached": True,
        },
    }
    monkeypatch.setitem(__import__("sys").modules, "requests", Requests)
    monkeypatch.setattr(_remote_http.time, "sleep", lambda _: None)
    monkeypatch.setattr(extract_remote, "unpack_and_verify", lambda *a, **k: envelope)
    assert extract_remote.main([
        "--job-id", job_id, "--out-dir", str(tmp_path), "--quiet",
    ]) == 0
    captured = capsys.readouterr()
    assert captured.out == "result"
    assert job_id in captured.err
    assert "OCR unavailable" in captured.err
    assert "Word alignment failed" in captured.err
    assert "Music detection failed" in captured.err
    assert "Frame cap reached" in captured.err
    assert sum(call.endswith("/bundle") for call in calls) == 3


def test_extraction_stream_honors_bundle_deadline(monkeypatch, tmp_path, capsys):
    now = [0.0]

    class BundleResponse:
        status_code = 200
        ok = True
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @staticmethod
        def iter_content(chunk_size):
            now[0] = 6.0
            yield b"late"

    class Requests:
        RequestException = OSError
        ConnectionError = OSError
        Timeout = TimeoutError
        get = staticmethod(lambda *a, **k: BundleResponse())

    monkeypatch.setattr(extract_remote, "poll_until_done", lambda *a, **k: {})
    monkeypatch.setattr(extract_remote.time, "monotonic", lambda: now[0])
    args = SimpleNamespace(poll=0.1, timeout=5.0, out_dir=str(tmp_path))
    assert extract_remote._fetch_extraction(
        Requests, "https://example", {}, "0123456789ab", args, lambda _: None,
    ) == 1
    error = capsys.readouterr().err
    assert "0123456789ab" in error
    assert "timed out" in error


def test_extraction_temp_write_error_is_clean(monkeypatch, tmp_path, capsys):
    class RequestError(Exception):
        pass

    class BundleResponse:
        status_code = 200
        ok = True
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @staticmethod
        def iter_content(chunk_size):
            yield b"data"

    class BrokenTemp:
        name = str(tmp_path / "bundle.tmp")

        @staticmethod
        def seek(*args):
            pass

        @staticmethod
        def truncate(*args):
            pass

        @staticmethod
        def close():
            pass

        @staticmethod
        def write(chunk):
            raise OSError("disk full")

    class Requests:
        RequestException = RequestError
        ConnectionError = RequestError
        Timeout = TimeoutError
        get = staticmethod(lambda *a, **k: BundleResponse())

    monkeypatch.setattr(extract_remote, "poll_until_done", lambda *a, **k: {})
    monkeypatch.setattr(extract_remote.tempfile, "NamedTemporaryFile", lambda **k: BrokenTemp())
    args = SimpleNamespace(poll=0.1, timeout=5.0, out_dir=str(tmp_path))
    assert extract_remote._fetch_extraction(
        Requests, "https://example", {}, "0123456789ab", args, lambda _: None,
    ) == 1
    error = capsys.readouterr().err
    assert "0123456789ab" in error
    assert "disk full" in error


def test_extraction_temp_creation_error_is_clean(monkeypatch, tmp_path, capsys):
    class Requests:
        RequestException = Exception

    def fail_temp(**kwargs):
        raise OSError("temp denied")

    monkeypatch.setattr(extract_remote, "poll_until_done", lambda *a, **k: {})
    monkeypatch.setattr(extract_remote.tempfile, "NamedTemporaryFile", fail_temp)
    args = SimpleNamespace(poll=0.1, timeout=5.0, out_dir=str(tmp_path))
    assert extract_remote._fetch_extraction(
        Requests, "https://example", {}, "0123456789ab", args, lambda _: None,
    ) == 1
    error = capsys.readouterr().err
    assert "0123456789ab" in error
    assert "temp denied" in error
