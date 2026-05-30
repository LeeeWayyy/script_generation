"""Normalize any media file into 16 kHz mono WAV, the format ASR models expect."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .ingest import ensure_tool

log = logging.getLogger("transcript.audio")

TARGET_SAMPLE_RATE = 16_000


def extract_audio(media_path: Path, work_dir: Path) -> Path:
    """Convert ``media_path`` (video or audio, any codec/container) to 16 kHz mono WAV.

    Uses ffmpeg, which must be installed on the system PATH.
    """
    ensure_tool("ffmpeg")
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / (media_path.stem + ".16k.wav")

    cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-i",
        str(media_path),
        "-vn",  # drop any video stream
        "-ac",
        "1",  # mono
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-acodec",
        "pcm_s16le",
        str(out_path),
    ]

    log.info("Extracting audio -> %s", out_path.name)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed to extract audio from {media_path}:\n{exc.stderr}") from exc

    return out_path
