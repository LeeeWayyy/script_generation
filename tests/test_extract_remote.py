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
    assert sniff_kind("https://x/clip.mp4?download=1#t=10") == "video"


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
            return {"id": "a1b2c3d4e5f6"}

    class _FakeRequests:
        RequestException = Exception

        @staticmethod
        def post(url, data=None, files=None, headers=None, timeout=None):
            captured["data"] = data
            captured["timeout"] = timeout
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
    assert captured["timeout"] == (30.0, 3600.0)


def test_audio_extraction_without_feed_or_enclosure_errors(monkeypatch, capsys):
    from transcript import extract_remote
    rc = extract_remote.main(["--kind", "audio_extraction"])  # no source/feed/enclosure
    assert rc == 1


def _envelope(**overrides) -> dict:
    return {"kind": "image_note", "text": "", "assets": [], **overrides}


def _bundle(envelope: dict, assets: dict[str, bytes], *, extra_members=None) -> bytes:
    envelope = _envelope(**envelope)
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
    env = {"assets": [{"key": "assets/a.jpg", "sha256": "0" * 64, "size": 3,
                       "media_type": "image/jpeg"}]}
    with pytest.raises(BundleVerificationError, match="sha256"):
        unpack_and_verify(_bundle(env, {"assets/a.jpg": b"abc"}), tmp_path / "o")


def test_unpack_rejects_size_mismatch(tmp_path):
    data = b"abc"
    env = {"assets": [{"key": "assets/a.jpg", "sha256": hashlib.sha256(data).hexdigest(),
                       "size": 99, "media_type": "image/jpeg"}]}
    with pytest.raises(BundleVerificationError, match="size"):
        unpack_and_verify(_bundle(env, {"assets/a.jpg": data}), tmp_path / "o")


def test_unpack_rejects_oversized_member_before_publish(tmp_path):
    data = b"x" * (2 * 1024 * 1024)
    env = {"assets": [{"key": "assets/a.jpg", "sha256": "0" * 64, "size": 1,
                       "media_type": "image/jpeg"}]}
    out = tmp_path / "o"
    with pytest.raises(BundleVerificationError, match="size"):
        unpack_and_verify(_bundle(env, {"assets/a.jpg": data}), out)
    assert not out.exists()
    assert not list(tmp_path.glob(".o-*"))


def test_unpack_rejects_negative_declared_size(tmp_path):
    env = {"assets": [{"key": "assets/a.jpg", "sha256": hashlib.sha256(b"").hexdigest(),
                       "size": -1,
                       "media_type": "image/jpeg"}]}
    with pytest.raises(BundleVerificationError, match="negative"):
        unpack_and_verify(_bundle(env, {"assets/a.jpg": b""}), tmp_path / "o")


def test_unpack_rejects_zip_slip_member(tmp_path):
    env = _envelope()
    with pytest.raises(BundleVerificationError):
        unpack_and_verify(_bundle(env, {}, extra_members={"../escape.txt": b"x"}),
                          tmp_path / "o")


def test_unpack_rejects_drive_letter_bundle_member(tmp_path):
    env = _envelope()
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


def test_unpack_rejects_unicode_normalization_collision(tmp_path):
    composed = "assets/é.txt"
    decomposed = "assets/e\u0301.txt"
    assets = {composed: b"first", decomposed: b"second"}
    env = _envelope(assets=[
        {
            "key": key,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
            "media_type": "text/plain",
        }
        for key, data in assets.items()
    ])
    with pytest.raises(BundleVerificationError, match="duplicate"):
        unpack_and_verify(_bundle(env, assets), tmp_path / "o")
    assert not (tmp_path / "o").exists()


def test_unpack_rejects_unicode_normalization_collision_in_asset_keys(tmp_path):
    composed = "assets/é.txt"
    data = b"same"
    meta = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "media_type": "text/plain",
    }
    env = _envelope(assets=[
        {"key": composed, **meta},
        {"key": "assets/e\u0301.txt", **meta},
    ])
    with pytest.raises(BundleVerificationError, match="duplicate asset key"):
        unpack_and_verify(_bundle(env, {composed: data}), tmp_path / "o")


@pytest.mark.parametrize(
    "key",
    [
        "assets/CON.txt",
        "assets/PRN .txt",
        "assets/AUX...txt",
        "assets/NUL.txt",
        "assets/CLOCK$.txt",
        "assets/COM1.txt",
        "assets/COM9.txt",
        "assets/LPT1.txt",
        "assets/LPT9.txt",
        "assets/CON/file.txt",
        "assets/file.txt:secret",
        "assets/x\x1f.txt",
        "assets/x\x7f.txt",
        "assets/bad<.txt",
        "assets/bad>.txt",
        'assets/bad".txt',
        "assets/bad|.txt",
        "assets/bad?.txt",
        "assets/bad*.txt",
        "assets/foo.",
        "assets/foo ",
    ],
)
def test_unpack_rejects_windows_unsafe_asset_components(tmp_path, key):
    data = b"x"
    env = _envelope(assets=[{
        "key": key,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "media_type": "text/plain",
    }])
    with pytest.raises(BundleVerificationError, match="Windows-unsafe"):
        unpack_and_verify(_bundle(env, {}), tmp_path / "o")


@pytest.mark.parametrize("name", ["assets/NUL.jpg", "assets/NUL/"])
def test_unpack_rejects_windows_unsafe_bundle_member(tmp_path, name):
    with pytest.raises(BundleVerificationError, match="Windows-unsafe"):
        unpack_and_verify(
            _bundle(_envelope(), {}, extra_members={name: b"x"}),
            tmp_path / "o",
        )


def test_unpack_rejects_windows_trailing_dot_alias_pair(tmp_path):
    data = b"x"
    assets = {"assets/foo": data, "assets/foo.": data}
    env = _envelope(assets=[{
        "key": key,
        "sha256": hashlib.sha256(value).hexdigest(),
        "size": len(value),
        "media_type": "application/octet-stream",
    } for key, value in assets.items()])
    with pytest.raises(BundleVerificationError, match="Windows-unsafe"):
        unpack_and_verify(_bundle(env, assets), tmp_path / "o")


def test_unpack_preflights_member_count_before_zipfile(monkeypatch, tmp_path):
    from transcript import extract_remote

    monkeypatch.setattr(extract_remote, "_MAX_MEMBERS", 1)
    data = b"x"
    env = _envelope(assets=[{
        "key": "assets/a.jpg",
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": 1,
        "media_type": "image/jpeg",
    }])
    with pytest.raises(BundleVerificationError, match="invalid zip bundle"):
        unpack_and_verify(_bundle(env, {"assets/a.jpg": data}), tmp_path / "o")


def test_unpack_failed_preflight_never_constructs_zipfile(monkeypatch, tmp_path):
    from transcript import extract_remote

    bundle = _bundle(_envelope(), {})

    def fail_preflight(*_args, **_kwargs):
        raise ValueError("bad central directory")

    def fail_zipfile(*_args, **_kwargs):
        raise AssertionError("ZipFile must not run after a failed preflight")

    monkeypatch.setattr(extract_remote, "preflight_zip", fail_preflight)
    monkeypatch.setattr(extract_remote.zipfile, "ZipFile", fail_zipfile)
    with pytest.raises(BundleVerificationError, match="bad central directory"):
        unpack_and_verify(bundle, tmp_path / "o")


def test_unpack_preflights_central_directory_bytes(monkeypatch, tmp_path):
    from transcript import extract_remote

    monkeypatch.setattr(extract_remote, "_MAX_CENTRAL_DIRECTORY_BYTES", 1)
    with pytest.raises(BundleVerificationError, match="invalid zip bundle"):
        unpack_and_verify(_bundle(_envelope(), {}), tmp_path / "o")


@pytest.mark.parametrize(
    ("encrypted", "external_attr", "message"),
    [(True, 0, "encrypted"), (False, 0o120777 << 16, "symlink")],
)
def test_member_flags_rejected_even_for_directory_entries(encrypted, external_attr, message):
    from transcript import extract_remote

    info = zipfile.ZipInfo("assets/")
    info.flag_bits = 1 if encrypted else 0
    info.external_attr = external_attr

    class FakeZip:
        @staticmethod
        def infolist():
            return [info]

    with pytest.raises(BundleVerificationError, match=message):
        extract_remote._safe_member_names(FakeZip())


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
        env = _envelope(assets=[
            {"key": bad, "sha256": "0", "size": 0, "media_type": "x"},
        ])
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("result.json", json.dumps(env))
        with pytest.raises(BundleVerificationError):
            unpack_and_verify(buf.getvalue(), tmp_path / "o")


def test_unpack_rejects_unsafe_asset_key_in_envelope(tmp_path):
    # The server-supplied envelope is untrusted: a `..` asset key must be rejected
    # before `out_dir / key` is ever used (zip members alone aren't enough).
    data = b"x"
    env = _envelope(assets=[{
        "key": "../escape.jpg", "sha256": hashlib.sha256(data).hexdigest(),
        "size": 1, "media_type": "image/jpeg",
    }])
    # Put the file under a benign member name so the zip itself is "safe".
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("result.json", json.dumps(env))
    with pytest.raises(BundleVerificationError, match="asset key"):
        unpack_and_verify(buf.getvalue(), tmp_path / "o")


def test_unpack_rejects_backslash_asset_key(tmp_path):
    data = b"x"
    env = _envelope(assets=[{
        "key": "assets\\card.jpg", "sha256": hashlib.sha256(data).hexdigest(),
        "size": 1, "media_type": "image/jpeg",
    }])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("result.json", json.dumps(env))
    with pytest.raises(BundleVerificationError, match="non-POSIX|asset key"):
        unpack_and_verify(buf.getvalue(), tmp_path / "o")


def test_unpack_rejects_unexpected_extra_member(tmp_path):
    # A member not in {result.json} ∪ {asset keys} must be rejected, and NOTHING
    # is written to disk (verify-before-write).
    env = _envelope()
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
    env = _envelope(assets=[
        {"key": "assets/a.jpg", "sha256": hashlib.sha256(good).hexdigest(),
         "size": 4, "media_type": "image/jpeg"},
        {"key": "assets/b.jpg", "sha256": "0" * 64, "size": 3,
         "media_type": "image/jpeg"},
    ])
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
    assert not list(tmp_path.glob(".o-*"))  # failed sibling staging was cleaned


def test_unpack_rejects_asset_key_claiming_result_json(tmp_path):
    # A compromised server can't list result.json as an asset key (reserved).
    env = _envelope(assets=[{
        "key": "result.json", "sha256": "0", "size": 0,
        "media_type": "application/json",
    }])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("result.json", json.dumps(env))
    with pytest.raises(BundleVerificationError, match="reserved"):
        unpack_and_verify(buf.getvalue(), tmp_path / "o")


def test_unpack_accepts_a_zip_path_and_streams(tmp_path):
    # The client downloads to a temp file and verifies from the PATH (bounded RAM).
    data = b"img"
    env = _envelope(text="hi", assets=[
        {"key": "assets/a.jpg", "sha256": hashlib.sha256(data).hexdigest(),
         "size": 3, "media_type": "image/jpeg"},
    ])
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
        zf.writestr("result.json", json.dumps(_envelope(
            assets=[{"key": "assets/a.jpg"}],
        )))
        zf.writestr("assets/a.jpg", b"x")
    with pytest.raises(BundleVerificationError, match="malformed"):
        unpack_and_verify(buf.getvalue(), tmp_path / "o")


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "unknown"},
        {"text": 123},
        {"assets": {}},
        {"assets": [{"key": "assets/a", "sha256": "nope", "size": 1}]},
        {"assets": [{"key": "assets/a", "sha256": "0" * 64, "size": 1}]},
    ],
)
def test_unpack_rejects_invalid_envelope_schema(tmp_path, bad):
    with pytest.raises(BundleVerificationError, match="malformed"):
        unpack_and_verify(_bundle(bad, {}), tmp_path / "o")


@pytest.mark.parametrize("error", [RuntimeError("encrypted"), NotImplementedError("codec")])
def test_unpack_wraps_result_read_errors(monkeypatch, tmp_path, error):
    real_read = zipfile.ZipFile.read

    def fail_result(self, name, *args, **kwargs):
        if name == "result.json":
            raise error
        return real_read(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", fail_result)
    with pytest.raises(BundleVerificationError, match="malformed"):
        unpack_and_verify(_bundle({}, {}), tmp_path / "o")


def test_unpack_wraps_unsupported_asset_open(monkeypatch, tmp_path):
    data = b"x"
    env = {"assets": [{
        "key": "assets/a.jpg", "sha256": hashlib.sha256(data).hexdigest(),
        "size": 1, "media_type": "image/jpeg",
    }]}
    bundle = _bundle(env, {"assets/a.jpg": data})
    real_open = zipfile.ZipFile.open

    def fail_asset(self, name, *args, **kwargs):
        if name == "assets/a.jpg":
            raise NotImplementedError("unsupported compression")
        return real_open(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", fail_asset)
    with pytest.raises(BundleVerificationError, match="failed writing"):
        unpack_and_verify(bundle, tmp_path / "o")
    assert not list(tmp_path.glob(".o-*"))


def test_unpack_requires_result_json(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("assets/a.jpg", b"x")
    with pytest.raises(BundleVerificationError, match="result.json"):
        unpack_and_verify(buf.getvalue(), tmp_path / "o")
