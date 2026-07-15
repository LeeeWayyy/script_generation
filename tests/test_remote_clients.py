"""Remote clients fail locally/cleanly and apply one timeout policy."""

import pytest

from transcript import extract_remote, remote
from transcript._remote_http import poll_until_done


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


def test_extraction_client_sends_speaker_and_music_flags(monkeypatch):
    captured = {}

    class Requests:
        RequestException = OSError

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
            "--detect-music",
            "--timeout", "12",
        ])
    assert captured["data"]["min_speakers"] == "2"
    assert captured["data"]["max_speakers"] == "4"
    assert captured["data"]["detect_music"] == "true"
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
