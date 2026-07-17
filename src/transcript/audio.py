"""Normalize any media file into 16 kHz mono WAV, the format ASR models expect."""

from __future__ import annotations

import logging
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path

from .ingest import _stop_process_tree, ensure_tool

log = logging.getLogger("transcript.audio")

TARGET_SAMPLE_RATE = 16_000
DEFAULT_MAX_AUDIO_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_AUDIO_TIMEOUT_S = 3600.0


def _audio_limits() -> tuple[int, float]:
    try:
        max_bytes = int(os.environ.get(
            "TRANSCRIPT_MAX_AUDIO_BYTES", DEFAULT_MAX_AUDIO_BYTES,
        ))
        timeout = float(os.environ.get(
            "TRANSCRIPT_AUDIO_TIMEOUT_S", DEFAULT_AUDIO_TIMEOUT_S,
        ))
    except ValueError as exc:
        raise RuntimeError("audio extraction limits must be numeric") from exc
    if max_bytes <= 0 or not math.isfinite(timeout) or timeout <= 0:
        raise RuntimeError("audio extraction limits must be finite and greater than zero")
    return max_bytes, timeout


def extract_audio(media_path: Path, work_dir: Path) -> Path:
    """Convert ``media_path`` (video or audio, any codec/container) to 16 kHz mono WAV.

    Uses ffmpeg, which must be installed on the system PATH.
    """
    ensure_tool("ffmpeg")
    max_bytes, timeout = _audio_limits()
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / (media_path.stem + ".16k.wav")
    with tempfile.TemporaryDirectory(
        prefix=".transcript-audio-", dir=work_dir, ignore_cleanup_errors=True,
    ) as temp_dir:
        temp_path = Path(temp_dir) / "output.wav"
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
            "-fs",
            str(max_bytes + 1),  # make truncation visible to the final > cap check
            str(temp_path),
        ]

        log.info("Extracting audio -> %s", out_path.name)
        popen_kwargs = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.PIPE,
            "text": True,
        }
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        elif os.name == "nt":
            popen_kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )

        def output_bytes() -> int:
            try:
                return temp_path.stat().st_size
            except FileNotFoundError:
                return 0

        deadline = time.monotonic() + timeout
        proc = None
        try:
            try:
                proc = subprocess.Popen(cmd, **popen_kwargs)
            except OSError as exc:
                raise RuntimeError(f"ffmpeg could not start: {exc}") from exc

            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"ffmpeg audio extraction timed out after {timeout:g}s"
                    )
                try:
                    _, stderr = proc.communicate(timeout=min(0.1, remaining))
                    break
                except subprocess.TimeoutExpired as exc:
                    if output_bytes() <= max_bytes:
                        continue
                    raise RuntimeError(
                        f"decoded audio exceeds the {max_bytes}-byte cap"
                    ) from exc

            if output_bytes() > max_bytes:
                raise RuntimeError(f"decoded audio exceeds the {max_bytes}-byte cap")
            if proc.returncode:
                raise RuntimeError(
                    f"ffmpeg failed to extract audio from {media_path}:\n{stderr}"
                )
            try:
                temp_path.replace(out_path)
            except OSError as exc:
                raise RuntimeError(f"could not finalize extracted audio: {exc}") from exc
        except BaseException:
            if proc is not None and proc.returncode is None:
                _stop_process_tree(proc)
            raise

    return out_path
