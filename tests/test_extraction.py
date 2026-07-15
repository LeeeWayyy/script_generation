"""Tests for the extraction envelope: renderers (pinned literal rules) + the one
explicit serializer (field order / inclusion policy / NFC). No ML deps."""

import hashlib
import json
import unicodedata
import zipfile

from transcript.extraction import (
    AssetRef,
    Card,
    ExtractionResult,
    Frame,
    render_audio_text,
    render_image_note_text,
    serialize,
    to_dict,
)
from transcript.types import Segment, Transcript, Word


# --- image_note text renderer: zero / one / many + trailing newline ---------


def test_image_note_zero_cards_is_empty_string():
    assert render_image_note_text([]) == ""


def _card(idx, ocr):
    return Card(index=idx, ocr_text=ocr, image_ref=f"assets/card-{idx:03d}.jpg",
                source_filename=f"{idx}.jpg", image_sha256="00")


def test_image_note_single_card_1_based_header_and_trailing_newline():
    out = render_image_note_text([_card(0, "Hello\nWorld")])
    # 1-based header even though index is 0-based; single trailing newline.
    assert out == "## card 1\nHello\nWorld\n"


def test_image_note_empty_ocr_card_is_header_only():
    out = render_image_note_text([_card(0, "")])
    assert out == "## card 1\n"  # header + the single trailing newline, no body


def test_image_note_many_cards_joined_by_blank_line():
    # Per the literal per-card rule, an empty card is "## card 1\n"; joined by
    # "\n\n" with the next non-empty card → three newlines between them.
    out = render_image_note_text([_card(0, ""), _card(1, "Hi")])
    assert out == "## card 1\n\n\n## card 2\nHi\n"


def test_image_note_trailing_empty_card_keeps_single_trailing_newline():
    out = render_image_note_text([_card(0, "Hi"), _card(1, "")])
    assert out == "## card 1\nHi\n\n## card 2\n"
    assert out.endswith("\n") and not out.endswith("\n\n")


def test_image_note_strips_trailing_crlf():
    out = render_image_note_text([_card(0, "Hello\r\n")])
    assert out == "## card 1\nHello\n"


def test_image_note_preserves_internal_blank_lines():
    # Internal OCR line breaks (incl. blank lines) are preserved verbatim; only
    # the single global trailing newline is normalized.
    out = render_image_note_text([_card(0, "a\n\nb"), _card(1, "c")])
    assert out == "## card 1\na\n\nb\n\n## card 2\nc\n"


def test_image_note_text_is_nfc_normalized():
    # Decomposed é (e + combining acute) must collapse to composed é.
    decomposed = "é"
    out = render_image_note_text([_card(0, decomposed)])
    assert "́" not in out
    assert out == unicodedata.normalize("NFC", out)


# --- audio/video text renderer = NFC(to_txt) --------------------------------


def test_render_audio_text_speaker_prefix_and_trailing_newline():
    t = Transcript(segments=[
        Segment(text="Hello there.", speaker="SPEAKER_00"),
        Segment(text="General Kenobi.", speaker="SPEAKER_01"),
        Segment(text="   ", speaker="SPEAKER_00"),  # blank → skipped
    ])
    out = render_audio_text(t)
    assert out == "SPEAKER_00: Hello there.\nSPEAKER_01: General Kenobi.\n"


def test_render_audio_text_is_nfc():
    t = Transcript(segments=[Segment(text="café")])
    out = render_audio_text(t)
    assert out == "café\n"


# --- serializer: field order, inclusion policy --------------------------------


def test_serialize_image_note_omits_segments_and_frames():
    r = ExtractionResult(
        kind="image_note",
        text="## card 1\nHi\n",
        language=None,
        cards=[_card(0, "Hi")],
        assets=[AssetRef(key="assets/card-000.jpg", sha256="ab", size=12, media_type="image/jpeg")],
        meta={"ocr_engine": "paddleocr@x"},
    )
    d = json.loads(serialize(r))
    assert list(d.keys()) == ["kind", "text", "language", "cards", "assets", "meta"]
    assert "segments" not in d and "frames" not in d
    assert d["assets"][0]["key"] == "assets/card-000.jpg"


def test_serialize_video_includes_segments_and_frames_even_when_empty():
    r = ExtractionResult(kind="video", text="\n", segments=[], frames=[], assets=[], meta={})
    d = json.loads(serialize(r))
    assert list(d.keys()) == ["kind", "text", "language", "segments", "frames", "assets", "meta"]
    assert d["segments"] == [] and d["frames"] == []


def test_serialize_is_deterministic_and_indent_2():
    r = ExtractionResult(kind="audio_extraction", text="hi\n", segments=[], meta={"a": 1})
    s1 = serialize(r)
    s2 = serialize(r)
    assert s1 == s2
    assert '\n  "kind"' in s1  # indent=2


def test_serialize_image_note_byte_golden():
    # Pin the EXACT serialized bytes of an image_note envelope so a field-order /
    # inclusion-policy / float-format change can't silently drift the contract
    # (the durable result.json must equal these bytes).
    r = ExtractionResult(
        kind="image_note", text="## card 1\nHello\n", language=None,
        cards=[Card(index=0, ocr_text="Hello", image_ref="assets/card-000.jpg",
                    source_filename="1.jpg", image_sha256="abc123",
                    width=100, height=50, confidence=0.9)],
        assets=[AssetRef(key="assets/card-000.jpg", sha256="def456", size=3,
                         media_type="image/jpeg")],
        meta={"ocr_model": "ch"},
    )
    golden = (
        '{\n'
        '  "kind": "image_note",\n'
        '  "text": "## card 1\\nHello\\n",\n'
        '  "language": null,\n'
        '  "cards": [\n'
        '    {\n'
        '      "index": 0,\n'
        '      "ocr_text": "Hello",\n'
        '      "image_ref": "assets/card-000.jpg",\n'
        '      "source_filename": "1.jpg",\n'
        '      "image_sha256": "abc123",\n'
        '      "width": 100,\n'
        '      "height": 50,\n'
        '      "confidence": 0.9\n'
        '    }\n'
        '  ],\n'
        '  "assets": [\n'
        '    {\n'
        '      "key": "assets/card-000.jpg",\n'
        '      "sha256": "def456",\n'
        '      "size": 3,\n'
        '      "media_type": "image/jpeg"\n'
        '    }\n'
        '  ],\n'
        '  "meta": {\n'
        '    "ocr_model": "ch"\n'
        '  }\n'
        '}'
    )
    assert serialize(r) == golden


def test_serialize_meta_preserves_insertion_order():
    r = ExtractionResult(kind="audio_extraction", text="x\n", segments=[],
                         meta={"feed_url": "u", "episode_guid": "g", "resolution_source": "feed_parse"})
    d = json.loads(serialize(r))
    assert list(d["meta"].keys()) == ["feed_url", "episode_guid", "resolution_source"]


def test_segment_word_field_order_pinned():
    r = ExtractionResult(
        kind="audio_extraction", text="hi\n",
        segments=[Segment(text="hi", start=0.0, end=1.0, speaker="S0",
                          words=[Word("hi", 0.0, 1.0, 0.9, "S0")])],
        meta={},
    )
    d = to_dict(r)
    assert list(d["segments"][0].keys()) == [
        "text", "start", "end", "speaker", "words", "music"
    ]
    assert d["segments"][0]["music"] is False
    assert list(d["segments"][0]["words"][0].keys()) == ["word", "start", "end", "score", "speaker"]


def test_frame_omits_optional_blocks_when_absent():
    r = ExtractionResult(kind="video", text="\n", segments=[],
                         frames=[Frame(frame_id=0, timecode=1.5, image_ref="assets/frame-000000.jpg")],
                         meta={})
    d = to_dict(r)
    f = d["frames"][0]
    assert list(f.keys()) == ["frame_id", "timecode", "image_ref", "ocr_text"]


def test_image_note_reuses_archive_member_hash_and_size(monkeypatch, tmp_path):
    from transcript.extract import extract_image_note
    from transcript.ocr import OcrResult

    data = b"already-hashed-original"
    archive = tmp_path / "cards.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("card.jpg", data)
    monkeypatch.setattr("transcript.ocr._load_engine", lambda: object())
    monkeypatch.setattr(
        "transcript.ocr.run_ocr",
        lambda *a, **k: OcrResult("text", 0.9, 10, 10, []),
    )
    monkeypatch.setattr(
        "transcript.extract._asset_ref",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("image was rehashed")),
    )

    result, _ = extract_image_note(archive, tmp_path / "assets")
    assert result.assets[0].sha256 == hashlib.sha256(data).hexdigest()
    assert result.assets[0].size == len(data)


def test_image_note_ocr_unavailable_is_attempted_once_and_recorded(
    monkeypatch, tmp_path, caplog,
):
    from transcript.extract import extract_image_note
    from transcript.ocr import OcrUnavailableError

    archive = tmp_path / "cards.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("one.jpg", b"one")
        zf.writestr("two.jpg", b"two")
    calls = 0

    def unavailable():
        nonlocal calls
        calls += 1
        raise OcrUnavailableError("weights missing at /private/server/path")

    monkeypatch.setattr("transcript.ocr._load_engine", unavailable)
    monkeypatch.setattr(
        "transcript.ocr.run_ocr",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("OCR retried")),
    )

    result, _ = extract_image_note(archive, tmp_path / "assets")
    assert calls == 1
    assert [card.ocr_text for card in result.cards] == ["", ""]
    assert len(result.assets) == 2
    assert result.meta["ocr_requested"] is True
    assert result.meta["ocr_succeeded"] is False
    assert "/private/server/path" not in result.meta["ocr_warning"]
    assert sum("OCR unavailable" in record.message for record in caplog.records) == 1


def test_video_ocr_unavailable_keeps_frames_and_does_not_retry(monkeypatch, tmp_path):
    from transcript.extract import extract_video
    from transcript.frames import FrameAsset
    from transcript.ocr import OcrUnavailableError

    frames = []
    for index in range(2):
        path = tmp_path / f"source-{index}.jpg"
        path.write_bytes(str(index).encode())
        frames.append(FrameAsset(index, float(index), path))
    monkeypatch.setattr("transcript.frames.extract_frames", lambda *a, **k: frames)
    monkeypatch.setattr("transcript._ffmpeg_version", lambda: "6.0")
    calls = 0

    def unavailable():
        nonlocal calls
        calls += 1
        raise OcrUnavailableError("missing")

    monkeypatch.setattr("transcript.ocr._load_engine", unavailable)
    result, _ = extract_video(
        transcript=Transcript(), video_path=tmp_path / "video.mp4",
        asset_dir=tmp_path / "assets",
    )
    assert calls == 1
    assert [frame.ocr_text for frame in result.frames] == ["", ""]
    assert len(result.assets) == 2
    assert result.meta["ocr_requested"] is True
    assert result.meta["ocr_succeeded"] is False
    assert "OCR unavailable" in result.meta["ocr_warning"]
