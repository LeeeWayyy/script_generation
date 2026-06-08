"""Thin extract-remote client: kind sniffing + bundle unpack/verify (zip-slip + integrity)."""

import hashlib
import io
import json
import zipfile

import pytest

from transcript.extract_remote import (BundleVerificationError, sniff_kind,
                                       unpack_and_verify)


def test_sniff_kind_by_extension():
    assert sniff_kind("export.zip") == "image_note"
    assert sniff_kind("https://x/clip.mp4") == "video"


def test_sniff_kind_plain_audio_directs_to_legacy():
    # Bare audio must NOT auto-route to audio_extraction (podcast-only).
    with pytest.raises(SystemExit, match="transcript-remote|audio_extraction"):
        sniff_kind("ep.mp3")


def test_sniff_kind_ambiguous_hard_fails():
    with pytest.raises(SystemExit):
        sniff_kind("mystery.bin")


def test_audio_extraction_positional_feed_url_is_routed(monkeypatch):
    # `extract-remote --kind audio_extraction <feed-url>` must route the positional
    # URL into feed_url (server ignores `url` for this kind) and submit — not 400.
    from transcript import extract_remote
    captured = {}

    class _Resp:
        status_code, ok = 200, True

        @staticmethod
        def json():
            return {"id": "job1"}

    class _FakeRequests:
        RequestException = Exception

        @staticmethod
        def post(url, data=None, files=None, headers=None):
            captured["data"] = data
            return _Resp()

    monkeypatch.setitem(__import__("sys").modules, "requests", _FakeRequests)
    # Short-circuit polling + bundle fetch so we only test submission.
    monkeypatch.setattr(extract_remote, "poll_until_done", lambda *a, **k: {"status": "done"})
    monkeypatch.setattr(extract_remote, "build_headers", lambda t: {})

    def _stop(*a, **k):
        raise SystemExit(0)
    monkeypatch.setattr(_FakeRequests, "get", _stop, raising=False)

    try:
        extract_remote.main(["--kind", "audio_extraction", "https://feed.example/rss",
                             "--episode-guid", "g1"])
    except SystemExit:
        pass
    assert captured["data"]["feed_url"] == "https://feed.example/rss"
    assert "url" not in captured["data"]  # not submitted as a plain url
    assert captured["data"]["episode_guid"] == "g1"


def test_audio_extraction_without_feed_or_enclosure_errors(monkeypatch, capsys):
    from transcript import extract_remote
    rc = extract_remote.main(["--kind", "audio_extraction"])  # no source/feed/enclosure
    assert rc == 1


def _bundle(envelope: dict, assets: dict[str, bytes], *, extra_members=None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("result.json", json.dumps(envelope))
        for key, data in assets.items():
            zf.writestr(key, data)
        for name, data in (extra_members or {}).items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_unpack_and_verify_happy_path(tmp_path):
    data = b"cardbytes"
    env = {"kind": "image_note", "text": "## card 1\nHi\n", "assets": [
        {"key": "assets/card-000.jpg", "sha256": hashlib.sha256(data).hexdigest(),
         "size": len(data), "media_type": "image/jpeg"}]}
    out = unpack_and_verify(_bundle(env, {"assets/card-000.jpg": data}), tmp_path / "o")
    assert out["text"] == "## card 1\nHi\n"
    assert (tmp_path / "o" / "assets" / "card-000.jpg").read_bytes() == data


def test_unpack_rejects_sha_mismatch(tmp_path):
    env = {"assets": [{"key": "assets/a.jpg", "sha256": "deadbeef", "size": 3,
                       "media_type": "image/jpeg"}]}
    with pytest.raises(BundleVerificationError, match="sha256"):
        unpack_and_verify(_bundle(env, {"assets/a.jpg": b"abc"}), tmp_path / "o")


def test_unpack_rejects_size_mismatch(tmp_path):
    data = b"abc"
    env = {"assets": [{"key": "assets/a.jpg", "sha256": hashlib.sha256(data).hexdigest(),
                       "size": 99, "media_type": "image/jpeg"}]}
    with pytest.raises(BundleVerificationError, match="size"):
        unpack_and_verify(_bundle(env, {"assets/a.jpg": data}), tmp_path / "o")


def test_unpack_rejects_zip_slip_member(tmp_path):
    env = {"assets": []}
    with pytest.raises(BundleVerificationError):
        unpack_and_verify(_bundle(env, {}, extra_members={"../escape.txt": b"x"}),
                          tmp_path / "o")


def test_unpack_rejects_drive_letter_bundle_member(tmp_path):
    env = {"assets": []}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("result.json", json.dumps(env))
        zf.writestr("C:/evil.txt", b"x")  # Windows drive-absolute member
    with pytest.raises(BundleVerificationError):
        unpack_and_verify(buf.getvalue(), tmp_path / "o")


def test_unpack_rejects_case_insensitive_duplicate_member(tmp_path):
    # a.jpg / A.jpg as distinct zip members collide on macOS/Windows → rejected.
    data = b"x"
    env = {"assets": [{"key": "assets/a.jpg", "sha256": hashlib.sha256(data).hexdigest(),
                       "size": 1, "media_type": "image/jpeg"}]}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("result.json", json.dumps(env))
        zf.writestr("assets/a.jpg", data)
        zf.writestr("assets/A.jpg", data)
    with pytest.raises(BundleVerificationError, match="duplicate"):
        unpack_and_verify(buf.getvalue(), tmp_path / "o")


def test_main_rejects_unsafe_server_job_id(monkeypatch, tmp_path):
    # A compromised server returning a path-like id must not be used as out_dir/id.
    from transcript import extract_remote

    class _Resp:
        status_code, ok = 200, True

        @staticmethod
        def json():
            return {"id": "../evil"}

    class _FakeRequests:
        RequestException = Exception

        @staticmethod
        def post(url, data=None, files=None, headers=None):
            return _Resp()

    monkeypatch.setitem(__import__("sys").modules, "requests", _FakeRequests)
    monkeypatch.setattr(extract_remote, "build_headers", lambda t: {})
    rc = extract_remote.main(["--kind", "audio_extraction", "--feed-url",
                              "https://feed/rss", "--out-dir", str(tmp_path)])
    assert rc == 1  # rejected, nothing written
    assert not (tmp_path / "..").joinpath("evil").exists()


def test_unpack_rejects_path_alias_asset_key(tmp_path):
    for bad in ("assets//a.jpg", "assets/./a.jpg"):
        env = {"assets": [{"key": bad, "sha256": "0", "size": 0, "media_type": "x"}]}
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("result.json", json.dumps(env))
        with pytest.raises(BundleVerificationError):
            unpack_and_verify(buf.getvalue(), tmp_path / "o")


def test_unpack_rejects_unsafe_asset_key_in_envelope(tmp_path):
    # The server-supplied envelope is untrusted: a `..` asset key must be rejected
    # before `out_dir / key` is ever used (zip members alone aren't enough).
    data = b"x"
    env = {"assets": [{"key": "../escape.jpg", "sha256": hashlib.sha256(data).hexdigest(),
                       "size": 1, "media_type": "image/jpeg"}]}
    # Put the file under a benign member name so the zip itself is "safe".
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("result.json", json.dumps(env))
    with pytest.raises(BundleVerificationError, match="asset key"):
        unpack_and_verify(buf.getvalue(), tmp_path / "o")


def test_unpack_rejects_backslash_asset_key(tmp_path):
    data = b"x"
    env = {"assets": [{"key": "assets\\card.jpg", "sha256": hashlib.sha256(data).hexdigest(),
                       "size": 1, "media_type": "image/jpeg"}]}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("result.json", json.dumps(env))
    with pytest.raises(BundleVerificationError, match="non-POSIX|asset key"):
        unpack_and_verify(buf.getvalue(), tmp_path / "o")


def test_unpack_rejects_unexpected_extra_member(tmp_path):
    # A member not in {result.json} ∪ {asset keys} must be rejected, and NOTHING
    # is written to disk (verify-before-write).
    env = {"assets": []}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("result.json", json.dumps(env))
        zf.writestr("sneaky.txt", b"unverified")  # not referenced by the envelope
    out = tmp_path / "o"
    with pytest.raises(BundleVerificationError, match="unexpected"):
        unpack_and_verify(buf.getvalue(), out)
    assert not (out / "sneaky.txt").exists()
    assert not (out / "result.json").exists()  # nothing written on failure


def test_unpack_nothing_written_until_all_assets_verified(tmp_path):
    good = b"good"
    env = {"assets": [
        {"key": "assets/a.jpg", "sha256": hashlib.sha256(good).hexdigest(),
         "size": 4, "media_type": "image/jpeg"},
        {"key": "assets/b.jpg", "sha256": "deadbeef", "size": 3, "media_type": "image/jpeg"},
    ]}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("result.json", json.dumps(env))
        zf.writestr("assets/a.jpg", good)
        zf.writestr("assets/b.jpg", b"bad")  # sha mismatch
    out = tmp_path / "o"
    with pytest.raises(BundleVerificationError, match="sha256"):
        unpack_and_verify(buf.getvalue(), out)
    # Even the valid asset must not have been written (atomic-ish verify-then-write).
    assert not (out / "assets" / "a.jpg").exists()


def test_unpack_rejects_asset_key_claiming_result_json(tmp_path):
    # A compromised server can't list result.json as an asset key (reserved).
    env = {"assets": [{"key": "result.json", "sha256": "0", "size": 0,
                       "media_type": "application/json"}]}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("result.json", json.dumps(env))
    with pytest.raises(BundleVerificationError, match="reserved"):
        unpack_and_verify(buf.getvalue(), tmp_path / "o")


def test_unpack_accepts_a_zip_path_and_streams(tmp_path):
    # The client downloads to a temp file and verifies from the PATH (bounded RAM).
    data = b"img"
    env = {"text": "hi", "assets": [
        {"key": "assets/a.jpg", "sha256": hashlib.sha256(data).hexdigest(),
         "size": 3, "media_type": "image/jpeg"}]}
    zpath = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("result.json", json.dumps(env))
        zf.writestr("assets/a.jpg", data)
    out = unpack_and_verify(zpath, tmp_path / "o")  # pass a Path, not bytes
    assert out["text"] == "hi"
    assert (tmp_path / "o" / "assets" / "a.jpg").read_bytes() == data


def test_unpack_rejects_malformed_envelope(tmp_path):
    # A result.json missing required asset fields is a verification failure, not
    # an uncaught KeyError.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("result.json", json.dumps({"assets": [{"key": "assets/a.jpg"}]}))
        zf.writestr("assets/a.jpg", b"x")
    with pytest.raises(BundleVerificationError, match="malformed"):
        unpack_and_verify(buf.getvalue(), tmp_path / "o")


def test_unpack_requires_result_json(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("assets/a.jpg", b"x")
    with pytest.raises(BundleVerificationError, match="result.json"):
        unpack_and_verify(buf.getvalue(), tmp_path / "o")
