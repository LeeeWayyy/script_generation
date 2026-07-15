"""Safe extraction of an untrusted manual-export bundle (zip/tar) — plan §Interface.

The ``image_note`` manual-export input is a zip or tar uploaded as one
``UploadFile``. Untrusted archives are a security minefield, so this module:

* rejects absolute / ``..`` members (zip-slip / tar path-traversal);
* rejects symlinks, tar hardlinks, and special files (block/char devices,
  FIFOs) — separate traversal/DoS vectors;
* flat-extracts by basename only (collision rule below);
* caps decompression (per-archive uncompressed-size + member-count limits) so a
  zip bomb can't fill the now-non-self-cleaning durable asset disk.

Card ordering — which drives ``index`` which drives the hashed ``text`` — is a
**UTF-8-byte sort over NFC-normalized basenames** (codepoint and byte order
differ for non-ASCII; a locale/case-sensitive sort drifts between a macOS client
and a Linux server). Basename collisions are **rejected**, never silently
overwritten (a lost member would shift every ``index``).

Pure stdlib (zipfile/tarfile/hashlib/unicodedata). Image *decoding* lives in
``ocr.py``; this module only safely lands original member bytes on disk.
"""

from __future__ import annotations

import hashlib
import posixpath
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .types import (
    has_windows_drive_prefix,
    is_windows_reserved_basename,
    nfc as _nfc,
)
from .zip_safety import preflight_zip

# Conservative defaults; a zip bomb or a runaway export trips these clearly.
MAX_TOTAL_UNCOMPRESSED = 2 * 1024 * 1024 * 1024  # 2 GiB across the whole archive
MAX_MEMBERS = 10_000
MAX_CENTRAL_DIRECTORY_BYTES = 64 * 1024 * 1024  # bound ZipFile's metadata allocation

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


class UnsafeArchiveError(Exception):
    """Raised when an archive member violates a hard safety rule."""


@dataclass
class ExtractedMember:
    """One safely-extracted image: original bytes landed at ``path``."""

    basename: str  # bare basename, NFC-normalized
    source_member: str  # sanitized original member path (debug observation only)
    path: Path  # where the original bytes were written (flat-extract)
    sha256: str  # over the ORIGINAL member bytes
    size: int


def _reject_path(name: str) -> None:
    p = name.replace("\\", "/")
    if not p:
        return
    # Validate even directory members (trailing slash) so an absolute / `..`
    # directory entry trips the rule rather than being silently skipped. The
    # caller decides separately (via is_dir()) not to extract directories.
    stripped = p.rstrip("/")
    # Reject POSIX-absolute AND Windows drive paths — both absolute ("C:/x") and
    # drive-RELATIVE ("C:x", which still escapes the dest on a Windows host). The
    # latter is not is_absolute() on Linux/macOS, so check it explicitly.
    if p.startswith("/") or Path(stripped).is_absolute() or has_windows_drive_prefix(stripped):
        raise UnsafeArchiveError(f"absolute member path rejected: {name!r}")
    if ".." in stripped.split("/"):
        raise UnsafeArchiveError(f"path-traversal member rejected: {name!r}")


def _is_image(basename: str) -> bool:
    return Path(basename).suffix.lower() in IMAGE_SUFFIXES


def extract_images(archive_path: Path, dest_dir: Path) -> list[ExtractedMember]:
    """Safely flat-extract image members from a zip/tar into ``dest_dir``.

    Returns members sorted by the §A ordering rule (UTF-8-byte sort over
    NFC-normalized basenames). Raises :class:`UnsafeArchiveError` on any unsafe
    member, a basename collision, or a tripped decompression cap.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        members = _extract_zip(archive_path, dest_dir)
    elif tarfile.is_tarfile(archive_path):
        members = _extract_tar(archive_path, dest_dir)
    else:
        raise UnsafeArchiveError("input is neither a valid zip nor tar archive")

    # Pin ordering: UTF-8-byte sort over the (already NFC) basenames.
    members.sort(key=lambda m: m.basename.encode("utf-8"))
    return members


def _register(seen: dict, basename: str, source_member: str) -> str:
    """NFC-normalize, enforce image-only, and reject basename collisions.

    Collisions are detected CASE-INSENSITIVELY (NFC + casefold): on a
    case-insensitive filesystem (macOS/Windows) ``a.jpg`` and ``A.jpg`` would
    flat-extract to the same file, silently overwriting and shifting every card
    ``index`` — so we reject the pair regardless of host.
    """
    nfc = _nfc(basename)
    if has_windows_drive_prefix(nfc):
        raise UnsafeArchiveError(f"drive-relative basename rejected: {basename!r}")
    if not _is_image(nfc):
        return ""  # non-image member: skip silently (exports carry stray files)
    if ":" in nfc or any(ord(char) < 32 or ord(char) == 127 for char in nfc):
        raise UnsafeArchiveError(
            f"Windows-unsafe image basename rejected: {nfc!r}"
        )
    if is_windows_reserved_basename(nfc):
        raise UnsafeArchiveError(
            f"Windows reserved device basename rejected: {nfc!r}"
        )
    folded = nfc.casefold()
    if folded in seen:
        raise UnsafeArchiveError(
            f"basename collision after flat-extract: {nfc!r} "
            f"(from {seen[folded]!r} and {source_member!r}) — rejected to keep "
            f"card index stable"
        )
    seen[folded] = source_member
    return nfc


# Stream members in bounded chunks so a single huge (or lying-header) member can
# never be read wholly into RAM — a zip/tar-bomb DoS the disk-size cap alone
# would not stop.
_CHUNK = 1024 * 1024  # 1 MiB


def _extract_zip(archive_path: Path, dest_dir: Path) -> list[ExtractedMember]:
    out: list[ExtractedMember] = []
    seen: dict[str, str] = {}
    budget = _Budget(MAX_TOTAL_UNCOMPRESSED)
    try:
        with archive_path.open("rb") as fh:
            preflight_zip(
                fh,
                max_entries=MAX_MEMBERS,
                max_central_directory_bytes=MAX_CENTRAL_DIRECTORY_BYTES,
            )
    except (OSError, ValueError) as exc:
        raise UnsafeArchiveError(f"unsafe ZIP directory: {exc}") from exc
    with zipfile.ZipFile(archive_path) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_MEMBERS:
            raise UnsafeArchiveError(f"archive has too many members (> {MAX_MEMBERS})")
        for info in infos:
            name = info.filename
            # Validate the path even for directory members so an absolute / `..`
            # directory entry trips the rule (contract: reject, don't silently skip).
            _reject_path(name)
            if info.is_dir():
                continue
            # Zip carries the Unix file type in the high bits of external_attr.
            # Reject EVERY non-regular type (symlink, FIFO, char/block device,
            # socket). A mode of 0 is a plain DOS entry (no Unix metadata).
            mode = (info.external_attr >> 16) & 0o170000
            if mode not in (0, 0o100000):  # regular file only
                raise UnsafeArchiveError(
                    f"non-regular zip member rejected (mode {oct(mode)}): {name!r}"
                )
            # posixpath (NOT Path/os.path): member names are platform-neutral data
            # already normalized to "/", and ntpath.basename would treat a leading
            # "X:" as a drive and corrupt a legit POSIX name like "0:00.jpg".
            basename = _register(seen, posixpath.basename(name.replace("\\", "/")), name)
            if not basename:
                continue
            with zf.open(info) as fh:
                out.append(_stream_land(dest_dir, basename, name, fh, budget))
    return out


def _extract_tar(archive_path: Path, dest_dir: Path) -> list[ExtractedMember]:
    out: list[ExtractedMember] = []
    seen: dict[str, str] = {}
    budget = _Budget(MAX_TOTAL_UNCOMPRESSED)
    count = 0
    with tarfile.open(archive_path) as tf:
        for member in tf:
            count += 1
            if count > MAX_MEMBERS:
                raise UnsafeArchiveError(f"archive has too many members (> {MAX_MEMBERS})")
            name = member.name
            _reject_path(name)  # validate before the dir-skip (see zip note above)
            if member.isdir():
                continue
            # Reject every non-regular-file member: symlinks, hardlinks, devices, FIFOs.
            if not member.isfile():
                raise UnsafeArchiveError(
                    f"non-regular tar member rejected ({_tar_kind(member)}): {name!r}"
                )
            # posixpath (NOT Path/os.path): see the zip path's note — ntpath.basename
            # would corrupt a legit POSIX name like "0:00.jpg" on a Windows host.
            basename = _register(seen, posixpath.basename(name.replace("\\", "/")), name)
            if not basename:
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            with fh:
                out.append(_stream_land(dest_dir, basename, name, fh, budget))
    return out


class _Budget:
    """A running uncompressed-bytes budget shared across an archive's members."""

    def __init__(self, cap: int):
        self.cap = cap
        self.used = 0

    def take(self, n: int) -> None:
        self.used += n
        if self.used > self.cap:
            raise UnsafeArchiveError(
                "uncompressed size cap exceeded (possible archive bomb)"
            )


def _tar_kind(member: tarfile.TarInfo) -> str:
    if member.issym():
        return "symlink"
    if member.islnk():
        return "hardlink"
    if member.ischr() or member.isblk():
        return "device"
    if member.isfifo():
        return "fifo"
    return "special"


def _stream_land(dest_dir: Path, basename: str, source_member: str, src_fh,
                 budget: "_Budget") -> ExtractedMember:
    """Stream a member's bytes to disk in bounded chunks, hashing as we go and
    charging the shared budget — so a single member never lands wholly in RAM and
    a lying header trips the cap mid-stream."""
    dest = dest_dir / basename
    hasher = hashlib.sha256()
    size = 0
    with dest.open("wb") as out:
        while True:
            chunk = src_fh.read(_CHUNK)
            if not chunk:
                break
            budget.take(len(chunk))
            hasher.update(chunk)
            out.write(chunk)
            size += len(chunk)
    return ExtractedMember(
        basename=basename,
        source_member=source_member.replace("\\", "/"),
        path=dest,
        sha256=hasher.hexdigest(),
        size=size,
    )
