"""Pure OCR ordering/confidence + frame cadence/naming logic. No heavy deps."""

from transcript.frames import frame_name, parse_showinfo_pts, round_timecode
from transcript.ocr import (aggregate_confidence, blocks_to_text,
                            sort_reading_order)


# --- OCR reading order + confidence -----------------------------------------


def _b(text, x0, y0, x1, y1, score=0.9):
    return {"text": text, "bbox": [x0, y0, x1, y1], "score": score}


def test_reading_order_top_to_bottom_then_left_to_right():
    blocks = [
        _b("bottom-right", 100, 200, 200, 230),
        _b("top-left", 0, 0, 50, 30),
        _b("top-right", 100, 0, 200, 30),
        _b("bottom-left", 0, 200, 50, 230),
    ]
    out = blocks_to_text(sort_reading_order(blocks))
    assert out == "top-left\ntop-right\nbottom-left\nbottom-right"


def test_reading_order_groups_same_line_by_vertical_overlap():
    # Slight y jitter within one line must still read left-to-right.
    blocks = [_b("world", 120, 2, 200, 32), _b("hello", 0, 0, 100, 30)]
    assert blocks_to_text(sort_reading_order(blocks)) == "hello\nworld"


def test_confidence_is_mean_of_nonzero_scores():
    assert aggregate_confidence([0.8, 1.0, 0.0]) == 0.9
    assert aggregate_confidence([]) is None
    assert aggregate_confidence([0.0, 0.0]) is None


def test_empty_blocks_render_empty():
    assert sort_reading_order([]) == []
    assert blocks_to_text([]) == ""


# --- OCR no-surprise-download guard -----------------------------------------


def test_ocr_default_deny_without_config(monkeypatch):
    import pytest
    from transcript.ocr import OcrUnavailableError, _load_engine
    monkeypatch.delenv("TRANSCRIPT_OCR_MODEL_DIR", raising=False)
    monkeypatch.delenv("TRANSCRIPT_OCR_ALLOW_DOWNLOAD", raising=False)
    with pytest.raises(OcrUnavailableError, match="not pinned|TRANSCRIPT_OCR"):
        _load_engine()


def test_ocr_missing_model_dir_fails_clearly(monkeypatch, tmp_path):
    import pytest
    from transcript.ocr import OcrUnavailableError, _load_engine
    monkeypatch.setenv("TRANSCRIPT_OCR_MODEL_DIR", str(tmp_path / "nope"))
    with pytest.raises(OcrUnavailableError, match="does not"):
        _load_engine()


def test_ocr_model_dir_missing_subdirs_fails_clearly(monkeypatch, tmp_path):
    import pytest
    from transcript.ocr import OcrUnavailableError, _load_engine
    (tmp_path / "weights").mkdir()  # exists but lacks det/rec/cls subdirs
    monkeypatch.setenv("TRANSCRIPT_OCR_MODEL_DIR", str(tmp_path / "weights"))
    with pytest.raises(OcrUnavailableError, match="missing|det|rec|cls"):
        _load_engine()


def test_ocr_native_import_failure_is_reported_as_unavailable(monkeypatch):
    import builtins
    import sys
    import types

    import pytest
    from transcript.ocr import OcrUnavailableError, _load_engine

    monkeypatch.delenv("TRANSCRIPT_OCR_MODEL_DIR", raising=False)
    monkeypatch.setenv("TRANSCRIPT_OCR_ALLOW_DOWNLOAD", "1")
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    real_import = builtins.__import__

    def broken_paddle(name, *args, **kwargs):
        if name == "paddleocr":
            raise OSError("native library failed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_paddle)
    with pytest.raises(OcrUnavailableError, match="failed to import"):
        _load_engine()


def test_ocr_pixel_cap_is_checked_before_decode_or_copy(monkeypatch, tmp_path):
    import pytest

    Image = pytest.importorskip("PIL.Image")
    ImageOps = pytest.importorskip("PIL.ImageOps")
    from transcript.ocr import _decode_image

    path = tmp_path / "large.png"
    Image.new("RGB", (10, 10)).save(path)
    monkeypatch.setenv("TRANSCRIPT_MAX_IMAGE_PIXELS", "99")
    monkeypatch.setattr(
        ImageOps, "exif_transpose",
        lambda *_: (_ for _ in ()).throw(AssertionError("decode started")),
    )
    with pytest.raises(ValueError, match="100 decoded pixels"):
        _decode_image(path)


# --- frame cadence / naming --------------------------------------------------


def test_frame_name_zero_padded():
    assert frame_name(0) == "frame-000000.jpg"
    assert frame_name(123) == "frame-000123.jpg"


def test_round_timecode_3dp():
    assert round_timecode(1.23456) == 1.235


def test_extract_frames_rejects_below_cadence_floor(tmp_path):
    import pytest
    from transcript.frames import MIN_CADENCE_S, extract_frames
    with pytest.raises(ValueError, match="cadence_s"):
        extract_frames(tmp_path / "v.mp4", tmp_path / "f", cadence_s=0.0)
    with pytest.raises(ValueError):
        extract_frames(tmp_path / "v.mp4", tmp_path / "f", cadence_s=MIN_CADENCE_S / 2)


def test_extract_frames_rejects_non_finite_cadence(tmp_path):
    import pytest
    from transcript.frames import extract_frames
    for cadence in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="cadence_s"):
            extract_frames(tmp_path / "v.mp4", tmp_path / "f", cadence_s=cadence)


def test_parse_showinfo_pts_extracts_source_timecodes_in_order():
    # Real per-frame timecodes come from ffmpeg showinfo's pts_time (video clock),
    # including a non-zero start offset — not a synthetic n*cadence grid.
    stderr = (
        "[Parsed_showinfo_1 @ 0x1] n:0 pts:90000 pts_time:1.5 pos:1 fmt:yuvj420p\n"
        "[Parsed_showinfo_1 @ 0x1] n:1 pts:540000 pts_time:6.5 pos:2 fmt:yuvj420p\n"
        "[Parsed_showinfo_1 @ 0x1] n:2 pts:990000 pts_time:11 pos:3 fmt:yuvj420p\n"
    )
    assert parse_showinfo_pts(stderr) == [1.5, 6.5, 11.0]
    assert parse_showinfo_pts("") == []


def test_parse_showinfo_pts_accepts_signed_and_scientific_values():
    stderr = "pts_time:-0.125 x\npts_time:+1e-3 x\npts_time:2.5E+1 x\n"
    assert parse_showinfo_pts(stderr) == [-0.125, 0.001, 25.0]


def test_ffmpeg_command_is_derived_from_policy_and_caps_dimensions(monkeypatch, tmp_path):
    import transcript.frames as frames

    captured = {}
    monkeypatch.setattr("transcript.ingest.ensure_tool", lambda _name: None)

    class Process:
        returncode = 0

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs

        def communicate(self, *, timeout):
            return None, ""

    monkeypatch.setattr(frames.subprocess, "Popen", Process)
    frames.extract_frames(tmp_path / "video.mp4", tmp_path / "out", cadence_s=7.5)

    cmd = captured["cmd"]
    vf = cmd[cmd.index("-vf") + 1]
    assert frames.FRAME_POLICY["selector"].format(cadence_s=7.5) in vf
    assert frames.FRAME_POLICY["scale"] in vf
    assert "1280" in vf and "720" in vf
    assert cmd[cmd.index("-vsync") + 1] == frames.FRAME_POLICY["vsync"]
    assert captured["kwargs"]["stderr"] is frames.subprocess.PIPE
    assert captured["kwargs"]["text"] is True


def test_frame_asset_cap_kills_ffmpeg_and_cleans_partial_frames(monkeypatch, tmp_path):
    import subprocess

    import pytest

    import transcript.frames as frames
    from transcript.ingest import PROCESS_STOP_TIMEOUT_S

    out_dir = tmp_path / "out"
    captured = {}
    monkeypatch.setattr("transcript.ingest.ensure_tool", lambda _name: None)
    monkeypatch.setattr(frames, "MAX_TOTAL_ASSET_BYTES", 3)

    class Process:
        returncode = None
        pid = None
        stdout = None
        stderr = None

        def __init__(self, cmd, **kwargs):
            self.cmd = cmd
            self.killed = False
            self.stop_timeout = None
            captured["process"] = self

        def communicate(self, *, timeout):
            if self.killed:
                self.stop_timeout = timeout
                self.returncode = -9
                return None, ""
            (out_dir / "frame-000001.jpg").write_bytes(b"1234")
            raise subprocess.TimeoutExpired(self.cmd, timeout)

        def kill(self):
            self.killed = True

    monkeypatch.setattr(frames.subprocess, "Popen", Process)

    with pytest.raises(ValueError, match="3-byte total cap"):
        frames.extract_frames(tmp_path / "video.mp4", out_dir)

    process = captured["process"]
    assert process.killed is True
    assert process.stop_timeout == PROCESS_STOP_TIMEOUT_S
    assert list(out_dir.glob("frame-*.jpg")) == []
