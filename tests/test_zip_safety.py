"""Bounded ZIP metadata preflight before stdlib ZipFile allocation."""

import io
import struct
import zipfile

import pytest

from transcript.zip_safety import preflight_zip


def _zip(*names: str) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        for name in names:
            zf.writestr(name, b"")
    return out.getvalue()


def _zip64_empty() -> bytes:
    record = struct.pack(
        "<4sQ2H2I4Q",
        b"PK\x06\x06", 44, 45, 45, 0, 0, 0, 0, 0, 0,
    )
    locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, 0, 1)
    eocd = struct.pack(
        "<4s4H2IH",
        b"PK\x05\x06", 0, 0, 0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0,
    )
    return record + locator + eocd


def test_preflight_normal_standard_and_zip64():
    assert preflight_zip(
        io.BytesIO(_zip("a", "b")), max_entries=2, max_central_directory_bytes=1024,
    ) == 2
    assert preflight_zip(
        io.BytesIO(_zip64_empty()), max_entries=0, max_central_directory_bytes=0,
    ) == 0


def test_preflight_rejects_standard_entry_limit_and_falsified_low_count():
    data = _zip("a", "b")
    with pytest.raises(ValueError, match="too many"):
        preflight_zip(io.BytesIO(data), max_entries=1, max_central_directory_bytes=1024)

    forged = bytearray(data)
    eocd = forged.rfind(b"PK\x05\x06")
    struct.pack_into("<2H", forged, eocd + 8, 1, 1)
    with pytest.raises(ValueError, match="count mismatch"):
        preflight_zip(io.BytesIO(forged), max_entries=2, max_central_directory_bytes=1024)


def test_preflight_rejects_central_directory_byte_limit():
    with pytest.raises(ValueError, match="metadata size"):
        preflight_zip(
            io.BytesIO(_zip("a")), max_entries=10, max_central_directory_bytes=1,
        )


@pytest.mark.parametrize(
    "data",
    [
        b"not a zip",
        _zip("a")[:-1],
        struct.pack(
            "<4s4H2IH",
            b"PK\x05\x06", 0, 0, 0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0,
        ),
    ],
)
def test_preflight_rejects_malformed_or_truncated_metadata(data):
    with pytest.raises(ValueError):
        preflight_zip(io.BytesIO(data), max_entries=10, max_central_directory_bytes=1024)


def test_preflight_rejects_inconsistent_zip64_counts():
    data = bytearray(_zip64_empty())
    struct.pack_into("<2Q", data, 24, 1, 2)
    with pytest.raises(ValueError, match="inconsistent"):
        preflight_zip(io.BytesIO(data), max_entries=10, max_central_directory_bytes=1024)


def test_preflight_rejects_inconsistent_legacy_and_zip64_disk_numbers():
    data = bytearray(_zip64_empty())
    eocd = data.rfind(b"PK\x05\x06")
    struct.pack_into("<H", data, eocd + 4, 1)
    with pytest.raises(ValueError, match="inconsistent ZIP64 disk number"):
        preflight_zip(io.BytesIO(data), max_entries=10, max_central_directory_bytes=1024)
