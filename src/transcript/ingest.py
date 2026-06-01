"""Turn a *source* (local path or URL) into a local media file on disk.

- Local path: validated and returned as-is.
- URL: downloaded with yt-dlp (best available audio) into ``work_dir``.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("transcript.ingest")


def is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def resolve_source(source: str, work_dir: Path) -> Path:
    """Return a local file path for ``source``, downloading it first if it's a URL."""
    if is_url(source):
        return _download_url(source, work_dir)

    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Source is not a file: {path}")
    return path


def _download_url(url: str, work_dir: Path) -> Path:
    """Download best-audio for ``url`` using yt-dlp; return the downloaded file path."""
    work_dir.mkdir(parents=True, exist_ok=True)
    # %(id)s keeps the name predictable and filesystem-safe.
    out_template = str(work_dir / "%(id)s.%(ext)s")

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-f",
        "bestaudio/best",
        "--no-playlist",
        "-o",
        out_template,
        "--write-info-json",   # full provenance metadata → <id>.info.json (read by transcribe())
        "--print",
        "after_move:filepath",
        "--no-simulate",
        url,
    ]

    log.info("Downloading %s with yt-dlp ...", url)
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "yt-dlp is not available. Install it with `pip install yt-dlp`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"yt-dlp failed to download {url}:\n{exc.stderr}") from exc

    # The last non-empty stdout line is the final file path (after_move:filepath).
    printed = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    if printed and os.path.exists(printed[-1]):
        return Path(printed[-1])

    # Fallback: pick the newest file in work_dir.
    candidates = sorted(work_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]

    raise RuntimeError(f"Download appeared to succeed but no file was found in {work_dir}.")


def ensure_tool(name: str) -> None:
    """Raise a friendly error if a required system binary is missing."""
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required tool '{name}' was not found on your PATH. "
            f"Install it (e.g. `brew install {name}` on macOS, or your OS package manager)."
        )
