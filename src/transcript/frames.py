"""Video frame extraction + (in the worker) per-frame OCR — plan §B.

Policy: **fixed cadence** (reproducible). The recipe records cadence, the exact
ffmpeg select/scale/colorspace, timestamp rounding, frame naming, and encoding
params — asset hashes are only useful if the recipe explains why they changed.
Frame *selection* is reproducible only against the recorded policy+ffmpeg
version; ``frames[].ocr_text`` is an **observation**, never recipe.

No cross-modal timestamp-alignment guarantee: frame timecodes are on the **video
stream clock** and transcript segment times are on the **ASR audio stream clock**
(plan §B) — consumers must not assume alignment. Frame OCR lives in ``frames[]``,
separate from the audio ``text``.

ffmpeg is invoked lazily (server-side only). The cadence math, timecode rounding,
and neutral naming are pure and unit-tested.
"""

from __future__ import annotations

import logging
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("transcript.frames")

# Pinned frame-encoding + selection recipe. Recorded on ExtractionResult.meta.
FRAME_POLICY = {
    "method": "fixed_cadence",
    "cadence_s": 5.0,  # one frame every N seconds (default)
    "frame_format": "jpg",
    "jpeg_quality": 2,  # ffmpeg -q:v (2 = high quality)
    "pixel_format": "yuvj420p",
    # Preserve aspect ratio, never upscale, and bound both landscape and portrait.
    "scale": (
        "w='min(1280,iw)':h='min(720,ih)':"
        "force_original_aspect_ratio=decrease:force_divisible_by=2"
    ),
    "scale_algorithm": "bicubic",
    "timecode_round_dp": 3,  # seconds, fixed to 3 decimal places
    "max_frames": 2000,  # frame cap for long video
    "exif_handling": "none",  # extracted frames carry no EXIF
    # The exact ffmpeg shape that selects frames + sources timestamps — recorded
    # so the recipe fully explains how frame assets/timecodes were produced.
    "selector": "select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{cadence_s})'",
    "vsync": "0",  # passthrough — preserves source PTS
    "timestamp_source": "showinfo:pts_time",  # frame timecodes come from showinfo
}

TIMECODE_ROUND_DP = FRAME_POLICY["timecode_round_dp"]

# Cadence floor (plan §B "cadence floor for long video"): below this, frame +
# OCR cost explodes, so reject rather than sample faster.
MIN_CADENCE_S = 0.5


@dataclass
class FrameAsset:
    """A frame extracted to disk; ``frame_id`` is the pinned ordinal N."""

    frame_id: int
    timecode: float  # seconds on the video stream clock, rounded
    path: Path


def frame_name(frame_id: int) -> str:
    """Neutral, zero-padded frame filename (pinned)."""
    return f"frame-{frame_id:06d}.jpg"


def round_timecode(seconds: float) -> float:
    """Round a frame timecode to the pinned precision (seconds, 3 dp)."""
    return round(seconds, TIMECODE_ROUND_DP)


def extract_frames(
    video_path: Path,
    dest_dir: Path,
    *,
    cadence_s: float = FRAME_POLICY["cadence_s"],
    max_frames: int = FRAME_POLICY["max_frames"],
) -> list[FrameAsset]:
    """Extract frames at a fixed cadence into ``dest_dir`` via ffmpeg.

    Returns the frames sorted by ``frame_id`` (== chronological). Timecodes come
    from ffmpeg ``showinfo`` source PTS, with the cadence grid used only as a
    fallback when a corresponding PTS is unavailable.
    """
    from .ingest import ensure_tool

    # A non-positive (or absurdly tiny) cadence would make the ffmpeg select filter
    # emit every frame and then run OCR on each — a resource-exhaustion vector.
    if not math.isfinite(cadence_s) or cadence_s < MIN_CADENCE_S:
        raise ValueError(f"cadence_s must be >= {MIN_CADENCE_S} (got {cadence_s})")
    ensure_tool("ffmpeg")
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(dest_dir / "frame-%06d.jpg")
    # Sample frames whose SOURCE timestamp (the video stream clock) is at least
    # `cadence_s` past the previously selected one — and read each selected frame's
    # real `pts_time` from showinfo. This is the video-stream clock (plan §B), so
    # it stays correct for non-zero start times, VFR inputs, and dropped frames —
    # unlike a synthetic n*cadence grid. `-vsync 0` (passthrough) keeps source PTS.
    select = FRAME_POLICY["selector"].format(cadence_s=cadence_s)
    scale = f"scale={FRAME_POLICY['scale']}:flags={FRAME_POLICY['scale_algorithm']}"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"{select},{scale},showinfo",
        "-vsync", FRAME_POLICY["vsync"],
        "-pix_fmt", FRAME_POLICY["pixel_format"],
        "-q:v", str(FRAME_POLICY["jpeg_quality"]),
        "-frames:v", str(max_frames),
        out_template,
    ]
    log.info("Extracting frames (cadence=%ss) from %s", cadence_s, video_path.name)
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed to extract frames:\n{exc.stderr}") from exc

    produced = sorted(dest_dir.glob("frame-*.jpg"))
    pts_times = parse_showinfo_pts(proc.stderr)
    frames: list[FrameAsset] = []
    for i, src in enumerate(produced):
        dest = dest_dir / frame_name(i)
        if src != dest:
            src.rename(dest)
        # Prefer the real per-frame PTS; fall back to the synthetic grid only if
        # showinfo parsing didn't line up (so we always return a timecode).
        tc = pts_times[i] if i < len(pts_times) else i * cadence_s
        frames.append(FrameAsset(frame_id=i, timecode=round_timecode(tc), path=dest))
    if len(frames) >= max_frames:
        log.warning("Frame cap (%d) reached; later frames were dropped.", max_frames)
    return frames


_PTS_RE = re.compile(
    r"pts_time:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


def parse_showinfo_pts(stderr: str) -> list[float]:
    """Extract per-frame ``pts_time`` values (source video stream clock, in order)
    from ffmpeg ``showinfo`` stderr — one entry per emitted frame."""
    return [float(m.group(1)) for m in _PTS_RE.finditer(stderr)]
