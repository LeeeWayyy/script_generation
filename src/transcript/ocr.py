"""OCR for ``image_note`` cards and video frames (plan §A).

Engine: PaddleOCR (strong CJK), **lazy-imported** so the thin client / tests
never pull it in. Model weights are expected to be **pre-fetched at deploy** to a
pinned cache path — a first-run download is a hidden network dependency that can
fail mid-job, so we fail clearly (cf. yt-dlp's friendly ``RuntimeError``) rather
than surprise-download.

Determinism is **soft**: identical only under a pinned engine+model+params+device
(NOT byte-stable across CPU/GPU, Paddle minor versions, or fp16 nondeterminism).
The mitigation is the recorded recipe — :data:`OCR_PARAMS` plus the engine/model
version — which lets the consumer *explain* why a re-run's ``ocr_text``
(an observation, never recipe) differs.

Reading order is **pinned here**, not taken from PaddleOCR's raw output order
(which varies with layout/model/orientation): :func:`sort_reading_order` defines
a top-to-bottom, then left-to-right line grouping, and ``ocr_text`` is generated
from that ordering. ``bbox`` is ``[x_min, y_min, x_max, y_max]`` in **pixel**
units, origin **top-left**.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# The pinned preprocessing + engine params — recorded on meta so divergent
# ``ocr_text`` across libraries/devices is explainable. OCR sees *decoded pixels*,
# so every step that changes pixels is here.
OCR_PARAMS = {
    "lang": "ch",  # PaddleOCR "ch" model covers Chinese + English (best CJK)
    "use_angle_cls": True,  # rotated-text angle classifier
    "det_limit_side_len": 960,  # detection resize cap (long side)
    "exif_transpose": True,  # honor EXIF orientation before OCR
    "colorspace": "RGB",  # decode/convert to RGB
    "alpha_flatten": "white",  # composite transparency over white
}

# A line whose vertical centre is within this fraction of the running line height
# is treated as the same text line (column/row grouping for multi-column + CJK).
_LINE_OVERLAP_RATIO = 0.5


class OcrUnavailableError(RuntimeError):
    """PaddleOCR (or its pre-fetched weights) is not available."""


@dataclass
class OcrResult:
    ocr_text: str
    confidence: Optional[float]
    width: Optional[int]
    height: Optional[int]
    blocks: list[dict]  # [{text, bbox:[x0,y0,x1,y1], score}] in reading order


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def aggregate_confidence(scores: list[float]) -> Optional[float]:
    """Per-card/-frame confidence = arithmetic mean of non-zero line scores.

    (Pinned aggregation rule — the field is otherwise unspecified.) ``None`` when
    there are no non-zero scores (e.g. an image with no detected text).
    """
    nonzero = [s for s in scores if s]
    if not nonzero:
        return None
    return sum(nonzero) / len(nonzero)


def sort_reading_order(blocks: list[dict]) -> list[dict]:
    """Sort detected blocks top-to-bottom, then left-to-right within a line.

    Blocks are grouped into lines by vertical-centre proximity (scaled by the
    running median line height) so multi-column / vertical-CJK / slightly-rotated
    captures order deterministically regardless of the engine's raw order.
    """
    if not blocks:
        return []

    def y0(b):
        return b["bbox"][1]

    def height(b):
        return max(1.0, b["bbox"][3] - b["bbox"][1])

    ordered = sorted(blocks, key=lambda b: (round(y0(b), 3), round(b["bbox"][0], 3)))
    lines: list[list[dict]] = []
    for b in ordered:
        placed = False
        for line in lines:
            # Anchor on the line's FIRST (leftmost/topmost) block, not the last —
            # a more stable line centre when a rightmost block is vertically offset.
            ref = line[0]
            ref_cy = (ref["bbox"][1] + ref["bbox"][3]) / 2
            b_cy = (b["bbox"][1] + b["bbox"][3]) / 2
            tol = _LINE_OVERLAP_RATIO * min(height(ref), height(b))
            if abs(b_cy - ref_cy) <= tol:
                line.append(b)
                placed = True
                break
        if not placed:
            lines.append([b])
    # Sort lines by their top edge, blocks within a line by their left edge.
    lines.sort(key=lambda ln: round(min(y0(x) for x in ln), 3))
    flat: list[dict] = []
    for line in lines:
        line.sort(key=lambda x: round(x["bbox"][0], 3))
        flat.extend(line)
    return flat


def blocks_to_text(blocks: list[dict]) -> str:
    """Join reading-ordered block texts (one per line). No trailing newline —
    the per-modality envelope renderer owns the final newline policy."""
    return "\n".join(b["text"] for b in blocks)


# ---------------------------------------------------------------------------
# The heavy path — lazy import, never at module top.
# ---------------------------------------------------------------------------


def _load_engine():
    import os

    # No-surprise-download guard (DEFAULT-DENY). Weights must be pre-fetched at
    # deploy to a pinned cache; a first-run download is a hidden network
    # dependency that can fail mid-job. So we refuse to construct PaddleOCR (which
    # may otherwise auto-download with default paths) UNLESS the operator either
    #   * points TRANSCRIPT_OCR_MODEL_DIR at the pre-fetched cache, or
    #   * explicitly opts in with TRANSCRIPT_OCR_ALLOW_DOWNLOAD=1.
    model_dir = os.environ.get("TRANSCRIPT_OCR_MODEL_DIR")
    allow_download = os.environ.get("TRANSCRIPT_OCR_ALLOW_DOWNLOAD") == "1"
    if model_dir and not os.path.isdir(model_dir):
        raise OcrUnavailableError(
            f"OCR model cache {model_dir!r} (TRANSCRIPT_OCR_MODEL_DIR) does not "
            f"exist. Pre-fetch the PaddleOCR weights there at deploy time."
        )
    if not model_dir and not allow_download:
        raise OcrUnavailableError(
            "OCR weights are not pinned. Set TRANSCRIPT_OCR_MODEL_DIR to a "
            "pre-fetched PaddleOCR weights dir, or set "
            "TRANSCRIPT_OCR_ALLOW_DOWNLOAD=1 to permit a first-run download. "
            "Refusing to surprise-download model files mid-job."
        )
    # Actually PIN the cache: point PaddleOCR's det/rec/cls model dirs at the
    # pre-fetched weights (laid out as <model_dir>/{det,rec,cls}) and require them
    # to be present (validated BEFORE importing PaddleOCR), so a misconfigured
    # cache fails clearly instead of letting PaddleOCR fall back to its own
    # (possibly-downloading) default.
    sub = {}
    if model_dir:
        sub = {f"{k}_model_dir": os.path.join(model_dir, k) for k in ("det", "rec", "cls")}
        missing = [d for d in sub.values() if not os.path.isdir(d)]
        if missing:
            raise OcrUnavailableError(
                f"Pinned OCR weights missing under {model_dir!r}: expected "
                f"{model_dir}/{{det,rec,cls}}. Pre-fetch them at deploy time."
            )
    # Import torch BEFORE paddleocr. On Windows, paddle/albumentations load native
    # DLLs that corrupt the loader's search state, so a later `import torch` fails
    # with "[WinError 127] ... shm.dll". Forcing torch's DLLs to load first makes
    # the order deterministic regardless of whether an ASR or OCR job ran first.
    # Harmless (and skipped) on a host where torch isn't installed.
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise OcrUnavailableError(
            "PaddleOCR is not installed. Install the server extra "
            "(`pip install -e \".[server]\"`) on the OCR host."
        ) from exc
    kwargs = dict(
        lang=OCR_PARAMS["lang"],
        use_angle_cls=OCR_PARAMS["use_angle_cls"],
        det_limit_side_len=OCR_PARAMS["det_limit_side_len"],
        show_log=False,
        **sub,
    )
    try:
        return PaddleOCR(**kwargs)
    except Exception as exc:  # noqa: BLE001 — weights missing / download blocked
        raise OcrUnavailableError(
            "PaddleOCR failed to initialize. Pre-fetch the model weights to the "
            "pinned cache (TRANSCRIPT_OCR_MODEL_DIR) at deploy time."
        ) from exc


def _decode_image(image_path: Path):
    from PIL import Image, ImageOps

    # Close the source descriptor promptly — Image.open is lazy and would
    # otherwise leak FDs across thousands of video frames. exif_transpose returns
    # the SAME object when there's no orientation tag, so detach with a copy (and
    # force pixels into memory) before the `with` closes the file handle.
    with Image.open(image_path) as src:
        img = ImageOps.exif_transpose(src) if OCR_PARAMS["exif_transpose"] else src
        if img is src:
            img = src.copy()
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def run_ocr(image_path: Path, engine=None) -> OcrResult:
    """OCR one image into an :class:`OcrResult` (reading-ordered).

    ``engine`` may be a pre-built PaddleOCR instance (reused across cards/frames
    so weights load once). Raises :class:`OcrUnavailableError` if OCR can't run.
    """
    import numpy as np

    eng = engine or _load_engine()
    img = _decode_image(image_path)
    width, height = img.size
    raw = eng.ocr(np.asarray(img), cls=OCR_PARAMS["use_angle_cls"])

    blocks: list[dict] = []
    # PaddleOCR returns [[ [box, (text, score)], ... ]] (one page).
    page = raw[0] if raw else []
    for entry in page or []:
        box, (text, score) = entry[0], entry[1]
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        blocks.append({
            "text": _nfc(text),
            "bbox": [min(xs), min(ys), max(xs), max(ys)],
            "score": float(score),
        })
    blocks = sort_reading_order(blocks)
    return OcrResult(
        ocr_text=blocks_to_text(blocks),
        confidence=aggregate_confidence([b["score"] for b in blocks]),
        width=width,
        height=height,
        blocks=blocks,
    )
