"""Video frame extraction + (in the worker) per-frame OCR — plan §B.

Policy: **fixed cadence is the DEFAULT** (reproducible); scene-detect is opt-in.
The recipe records cadence, the exact ffmpeg select/scale/colorspace, timestamp
rounding, frame naming, encoding params, and (when dedup is on) the pHash
impl+size+distance — asset hashes are only useful if the recipe explains why they
changed. Frame *selection* is reproducible only against the recorded
policy+ffmpeg version; ``frames[].ocr_text`` is an **observation**, never recipe.

No cross-modal timestamp-alignment guarantee: frame timecodes are on the **video
stream clock** and transcript segment times are on the **ASR audio stream clock**
(plan §B) — consumers must not assume alignment. Frame OCR lives in ``frames[]``,
separate from the audio ``text``.

ffmpeg is invoked lazily (server-side only). The cadence math, timecode rounding,
and neutral naming are pure and unit-tested.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger("transcript.frames")

# Pinned frame-encoding + selection recipe. Recorded on ExtractionResult.meta.
FRAME_POLICY = {
    "method": "fixed_cadence",
    "cadence_s": 5.0,  # one frame every N seconds (default)
    "frame_format": "jpg",
    "jpeg_quality": 2,  # ffmpeg -q:v (2 = high quality)
    "pixel_format": "yuvj420p",
    "scale": "iw:ih",  # no rescale by default
    "scale_algorithm": "bicubic",
    "timecode_round_dp": 3,  # seconds, fixed to 3 decimal places
    "max_frames": 2000,  # frame cap for long video
    "exif_handling": "none",  # extracted frames carry no EXIF
}

TIMECODE_ROUND_DP = FRAME_POLICY["timecode_round_dp"]


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


def planned_timecodes(duration_s: float, cadence_s: float, max_frames: int) -> list[float]:
    """The deterministic list of frame timecodes for a fixed-cadence pass.

    A frame at t=0, then every ``cadence_s`` up to ``duration_s``, capped at
    ``max_frames`` (the cadence floor for long video — we stop emitting rather
    than silently widen the interval, and the caller logs the cap).
    """
    if cadence_s <= 0:
        raise ValueError("cadence_s must be positive")
    times: list[float] = []
    t = 0.0
    while t <= duration_s + 1e-9 and len(times) < max_frames:
        times.append(round_timecode(t))
        t += cadence_s
    return times


def extract_frames(
    video_path: Path,
    dest_dir: Path,
    *,
    cadence_s: float = FRAME_POLICY["cadence_s"],
    max_frames: int = FRAME_POLICY["max_frames"],
    ffmpeg_version: Optional[str] = None,
) -> list[FrameAsset]:
    """Extract frames at a fixed cadence into ``dest_dir`` via ffmpeg.

    Returns the frames sorted by ``frame_id`` (== chronological). The frame
    timecodes are computed from the cadence (the ffmpeg ``fps`` filter emits on a
    regular grid), so naming/timecodes are reproducible against the recipe.
    """
    from .ingest import ensure_tool

    ensure_tool("ffmpeg")
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(dest_dir / "frame-%06d.jpg")
    # Sample frames whose SOURCE timestamp (the video stream clock) is at least
    # `cadence_s` past the previously selected one — and read each selected frame's
    # real `pts_time` from showinfo. This is the video-stream clock (plan §B), so
    # it stays correct for non-zero start times, VFR inputs, and dropped frames —
    # unlike a synthetic n*cadence grid. `-vsync 0` (passthrough) keeps source PTS.
    select = f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{cadence_s})'"
    scale = f"scale={FRAME_POLICY['scale']}:flags={FRAME_POLICY['scale_algorithm']}"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"{select},{scale},showinfo",
        "-vsync", "0",
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


_PTS_RE = re.compile(r"pts_time:\s*([0-9]+(?:\.[0-9]+)?)")


def parse_showinfo_pts(stderr: str) -> list[float]:
    """Extract per-frame ``pts_time`` values (source video stream clock, in order)
    from ffmpeg ``showinfo`` stderr — one entry per emitted frame."""
    return [float(m.group(1)) for m in _PTS_RE.finditer(stderr)]
