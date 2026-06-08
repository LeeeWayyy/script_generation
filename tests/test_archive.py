"""Safe untrusted-archive extraction + card ordering (plan §A / §Interface)."""

import io
import tarfile
import zipfile

import pytest

from transcript.archive import (MAX_MEMBERS, UnsafeArchiveError, extract_images)


def _zip(members: dict[str, bytes]) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    buf.seek(0)
    return buf


def _write_zip(tmp_path, members):
    p = tmp_path / "in.zip"
    p.write_bytes(_zip(members).getvalue())
    return p


def test_flat_extract_and_byte_sorted_ordering(tmp_path):
    # Nested dirs flatten to basenames; ordering is UTF-8-byte sort over NFC names.
    arc = _write_zip(tmp_path, {"b/2.jpg": b"two", "a/1.jpg": b"one", "z/10.png": b"ten"})
    members = extract_images(arc, tmp_path / "out")
    names = [m.basename for m in members]
    # Byte sort: "1.jpg" < "10.png" < "2.jpg"
    assert names == ["1.jpg", "10.png", "2.jpg"]
    assert all(m.path.is_file() for m in members)


def test_non_image_members_skipped(tmp_path):
    arc = _write_zip(tmp_path, {"a.jpg": b"x", "notes.txt": b"hi", "readme.md": b"y"})
    members = extract_images(arc, tmp_path / "out")
    assert [m.basename for m in members] == ["a.jpg"]


def test_basename_collision_rejected(tmp_path):
    arc = _write_zip(tmp_path, {"a/1.jpg": b"x", "b/1.jpg": b"y"})
    with pytest.raises(UnsafeArchiveError, match="collision"):
        extract_images(arc, tmp_path / "out")


def test_case_insensitive_basename_collision_rejected(tmp_path):
    # On macOS/Windows, a.jpg and A.jpg flat-extract to the same file → reject.
    arc = _write_zip(tmp_path, {"x/a.jpg": b"x", "y/A.jpg": b"y"})
    with pytest.raises(UnsafeArchiveError, match="collision"):
        extract_images(arc, tmp_path / "out")


def test_zip_slip_absolute_rejected(tmp_path):
    arc = _write_zip(tmp_path, {"/etc/evil.jpg": b"x"})
    with pytest.raises(UnsafeArchiveError):
        extract_images(arc, tmp_path / "out")


def test_zip_slip_dotdot_rejected(tmp_path):
    arc = _write_zip(tmp_path, {"../escape.jpg": b"x"})
    with pytest.raises(UnsafeArchiveError, match="traversal"):
        extract_images(arc, tmp_path / "out")


def test_sha256_over_original_bytes(tmp_path):
    import hashlib
    arc = _write_zip(tmp_path, {"a.jpg": b"hello"})
    [m] = extract_images(arc, tmp_path / "out")
    assert m.sha256 == hashlib.sha256(b"hello").hexdigest()
    assert m.size == 5


def test_member_count_cap(tmp_path):
    many = {f"{i}.jpg": b"x" for i in range(MAX_MEMBERS + 1)}
    arc = _write_zip(tmp_path, many)
    with pytest.raises(UnsafeArchiveError, match="too many"):
        extract_images(arc, tmp_path / "out")


def test_tar_symlink_rejected(tmp_path):
    p = tmp_path / "in.tar"
    with tarfile.open(p, "w") as tf:
        info = tarfile.TarInfo("link.jpg")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    with pytest.raises(UnsafeArchiveError, match="symlink"):
        extract_images(p, tmp_path / "out")


def test_tar_fifo_rejected(tmp_path):
    p = tmp_path / "in.tar"
    with tarfile.open(p, "w") as tf:
        info = tarfile.TarInfo("fifo.jpg")
        info.type = tarfile.FIFOTYPE
        tf.addfile(info)
    with pytest.raises(UnsafeArchiveError):
        extract_images(p, tmp_path / "out")


def test_streaming_cap_trips_on_actual_bytes(tmp_path, monkeypatch):
    # Lower the cap; a member exceeding it must be rejected while streaming,
    # without reading the whole member into memory first.
    import transcript.archive as arc
    monkeypatch.setattr(arc, "MAX_TOTAL_UNCOMPRESSED", 8)
    arc_path = _write_zip(tmp_path, {"big.jpg": b"x" * 64})
    with pytest.raises(UnsafeArchiveError, match="size cap"):
        arc.extract_images(arc_path, tmp_path / "out")


def test_unsafe_directory_member_rejected_not_skipped(tmp_path):
    # A `..` directory member must trip the rule, not be silently skipped.
    p = tmp_path / "in.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("../evil/", b"")  # directory entry with traversal
        zf.writestr("ok.jpg", b"x")
    with pytest.raises(UnsafeArchiveError, match="traversal"):
        extract_images(p, tmp_path / "out")


def test_zip_fifo_mode_rejected(tmp_path):
    # A zip member carrying a non-regular Unix mode (FIFO) in external_attr must
    # be rejected, mirroring the tar "regular files only" policy.
    p = tmp_path / "in.zip"
    with zipfile.ZipFile(p, "w") as zf:
        info = zipfile.ZipInfo("pipe.jpg")
        info.external_attr = 0o010000 << 16  # S_IFIFO
        zf.writestr(info, b"x")
    with pytest.raises(UnsafeArchiveError, match="non-regular"):
        extract_images(p, tmp_path / "out")


def test_zip_drive_letter_rejected_but_posix_colon_ok(tmp_path):
    # Both drive-ABSOLUTE ("C:/x") and drive-RELATIVE ("C:x") escape on Windows.
    for bad in ("C:/evil.jpg", "C:evil.jpg"):
        with pytest.raises(UnsafeArchiveError):
            extract_images(_write_zip(tmp_path, {bad: b"x"}), tmp_path / "o")
    # A POSIX filename whose colon is not a drive prefix is fine.
    members = extract_images(_write_zip(tmp_path, {"0:00.jpg": b"x"}), tmp_path / "ok")
    assert members[0].basename == "0:00.jpg"


def test_source_member_preserved_as_observation(tmp_path):
    arc = _write_zip(tmp_path, {"deep/nested/photo.jpg": b"x"})
    [m] = extract_images(arc, tmp_path / "out")
    assert m.source_member == "deep/nested/photo.jpg"
    assert m.basename == "photo.jpg"
