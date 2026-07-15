"""Bounded ZIP central-directory validation before ``zipfile`` allocates it."""

from __future__ import annotations

import struct

_EOCD = b"PK\x05\x06"
_EOCD64 = b"PK\x06\x06"
_EOCD64_LOCATOR = b"PK\x06\x07"
_CENTRAL = b"PK\x01\x02"
_EOCD_SIZE = 22
_EOCD64_SIZE = 56
_EOCD64_LOCATOR_SIZE = 20
_CENTRAL_SIZE = 46
_MAX_COMMENT = (1 << 16) - 1
_MAX_EOCD64_BYTES = 1024 * 1024


def _read_exact(fileobj, size: int) -> bytes:
    data = fileobj.read(size)
    if len(data) != size:
        raise ValueError("truncated ZIP structure")
    return data


def _find_eocd(fileobj, file_size: int):
    tail_size = min(file_size, _MAX_COMMENT + _EOCD_SIZE)
    fileobj.seek(file_size - tail_size)
    tail = _read_exact(fileobj, tail_size)
    end = len(tail)
    while True:
        offset = tail.rfind(_EOCD, 0, end)
        if offset < 0:
            raise ValueError("ZIP end-of-central-directory record not found")
        if offset + _EOCD_SIZE <= len(tail):
            fields = struct.unpack_from("<4s4H2LH", tail, offset)
            location = file_size - tail_size + offset
            if location + _EOCD_SIZE + fields[-1] == file_size:
                return fields, location
        end = offset


def _find_eocd64(fileobj, eocd_location: int):
    locator_location = eocd_location - _EOCD64_LOCATOR_SIZE
    if locator_location < 0:
        raise ValueError("ZIP64 locator is missing")
    fileobj.seek(locator_location)
    signature, disk, relative_offset, disks = struct.unpack(
        "<4sLQL", _read_exact(fileobj, _EOCD64_LOCATOR_SIZE)
    )
    if signature != _EOCD64_LOCATOR:
        raise ValueError("ZIP64 locator is missing")
    if disk != 0 or disks != 1:
        raise ValueError("multi-disk ZIP archives are not supported")

    scan_size = min(locator_location, _MAX_EOCD64_BYTES)
    fileobj.seek(locator_location - scan_size)
    tail = _read_exact(fileobj, scan_size)
    end = len(tail)
    while True:
        offset = tail.rfind(_EOCD64, 0, end)
        if offset < 0:
            raise ValueError("ZIP64 end-of-central-directory record not found")
        if offset + 12 <= len(tail):
            record_size = struct.unpack_from("<Q", tail, offset + 4)[0] + 12
            location = locator_location - scan_size + offset
            if (record_size >= _EOCD64_SIZE
                    and record_size <= _MAX_EOCD64_BYTES
                    and location + record_size == locator_location):
                if offset + _EOCD64_SIZE > len(tail):
                    raise ValueError("truncated ZIP64 end-of-central-directory record")
                fields = struct.unpack_from("<4sQ2H2L4Q", tail, offset)
                return fields, location, relative_offset
        end = offset


def _directory_layout(fileobj, file_size: int):
    eocd, eocd_location = _find_eocd(fileobj, file_size)
    (_signature, disk, directory_disk, entries_disk, entries_total,
     directory_size, directory_offset, _comment_size) = eocd
    needs_zip64 = (
        disk == 0xFFFF or directory_disk == 0xFFFF
        or entries_disk == 0xFFFF or entries_total == 0xFFFF
        or directory_size == 0xFFFFFFFF or directory_offset == 0xFFFFFFFF
    )
    if not needs_zip64:
        if disk != 0 or directory_disk != 0 or entries_disk != entries_total:
            raise ValueError("multi-disk or inconsistent ZIP directory")
        directory_start = eocd_location - directory_size
        concat = directory_start - directory_offset
        if directory_start < 0 or concat < 0:
            raise ValueError("invalid ZIP central-directory offset")
        return entries_total, directory_start, directory_size

    eocd64, eocd64_location, relative_offset = _find_eocd64(
        fileobj, eocd_location
    )
    (_signature, _record_size, _created, _needed, disk64, directory_disk64,
     entries_disk64, entries_total64, directory_size64, directory_offset64) = eocd64
    if disk64 != 0 or directory_disk64 != 0 or entries_disk64 != entries_total64:
        raise ValueError("multi-disk or inconsistent ZIP64 directory")
    for legacy, current, sentinel, label in (
        (disk, disk64, 0xFFFF, "disk number"),
        (directory_disk, directory_disk64, 0xFFFF, "directory disk"),
        (entries_disk, entries_disk64, 0xFFFF, "entry count"),
        (entries_total, entries_total64, 0xFFFF, "entry count"),
        (directory_size, directory_size64, 0xFFFFFFFF, "directory size"),
        (directory_offset, directory_offset64, 0xFFFFFFFF, "directory offset"),
    ):
        if legacy != sentinel and legacy != current:
            raise ValueError(f"inconsistent ZIP64 {label}")
    concat = eocd64_location - relative_offset
    directory_start = directory_offset64 + concat
    if (concat < 0 or directory_start < 0
            or directory_start + directory_size64 != eocd64_location):
        raise ValueError("invalid ZIP64 central-directory offset")
    return entries_total64, directory_start, directory_size64


def preflight_zip(fileobj, *, max_entries: int,
                  max_central_directory_bytes: int) -> int:
    """Validate and count a seekable ZIP's central records without allocating them."""
    if max_entries < 0 or max_central_directory_bytes < 0:
        raise ValueError("ZIP limits must be non-negative")
    try:
        original_position = fileobj.tell()
        fileobj.seek(0, 2)
        file_size = fileobj.tell()
        declared, directory_start, directory_size = _directory_layout(
            fileobj, file_size
        )
        if declared > max_entries:
            raise ValueError(f"ZIP has too many entries (> {max_entries})")
        if directory_size > max_central_directory_bytes:
            raise ValueError(
                "ZIP central directory exceeds the metadata size limit"
            )
        if directory_start + directory_size > file_size:
            raise ValueError("ZIP central directory extends past end of file")

        fileobj.seek(directory_start)
        remaining = directory_size
        actual = 0
        while remaining:
            if remaining < _CENTRAL_SIZE:
                raise ValueError("truncated ZIP central-directory record")
            header = _read_exact(fileobj, _CENTRAL_SIZE)
            if header[:4] != _CENTRAL:
                raise ValueError("invalid ZIP central-directory signature")
            name_size, extra_size, comment_size = struct.unpack_from("<3H", header, 28)
            variable_size = name_size + extra_size + comment_size
            record_size = _CENTRAL_SIZE + variable_size
            if record_size > remaining:
                raise ValueError("truncated ZIP central-directory record")
            fileobj.seek(variable_size, 1)
            remaining -= record_size
            actual += 1
            if actual > max_entries:
                raise ValueError(f"ZIP has too many entries (> {max_entries})")
        if actual != declared:
            raise ValueError(
                f"ZIP entry count mismatch (declared {declared}, found {actual})"
            )
        return actual
    except OSError as exc:
        raise ValueError(f"could not read ZIP structures: {exc}") from exc
    finally:
        try:
            fileobj.seek(original_position)
        except (OSError, UnboundLocalError):
            pass
