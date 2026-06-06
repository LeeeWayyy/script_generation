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


class SsrfError(ValueError):
    """A URL points at a non-public (private/loopback/link-local) host."""


def assert_public_url(url: str) -> None:
    """SSRF guard for a user-supplied fetch URL: require http(s) and reject a host
    that resolves to a private / loopback / link-local / reserved / multicast
    address (cloud metadata, localhost, the LAN). Opt out on a trusted intranet
    with TRANSCRIPT_ALLOW_PRIVATE_FETCH=1. (Residual DNS-rebinding TOCTOU and any
    redirects a downstream fetcher follows internally are out of scope here.)"""
    import ipaddress
    import socket
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise SsrfError(f"URL scheme not allowed: {url!r}")
    if os.environ.get("TRANSCRIPT_ALLOW_PRIVATE_FETCH") == "1":
        return
    host = parts.hostname
    if not host:
        raise SsrfError(f"no host in URL: {url!r}")
    try:
        infos = socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise SsrfError(f"cannot resolve host {host!r}: {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            raise SsrfError(f"refusing to fetch a non-public host: {host} → {ip}")


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
        "--",   # end of options: a URL starting with '-' can't be parsed as a flag
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

    # Fallback: pick the newest file in work_dir, excluding the sidecar
    # `<id>.info.json` (written by --write-info-json) — it is typically NEWER than
    # the media, so a naive newest-file pick would wrongly return the JSON.
    candidates = sorted(
        (p for p in work_dir.glob("*") if p.suffix.lower() != ".json"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if candidates:
        return candidates[0]

    raise RuntimeError(f"Download appeared to succeed but no file was found in {work_dir}.")


def download_frame_video(url: str, work_dir: Path, *, height_cap: int = 720) -> tuple[Path, str | None]:
    """Download a *capped video stream* for frame extraction (plan §B).

    SSRF note: the caller validates ``url``'s host via :func:`assert_public_url`
    at submit, but yt-dlp follows redirects with its own networking, so a public
    URL that *redirects* to a private host is a residual SSRF risk that can't be
    blocked here without reimplementing yt-dlp's transport. The deployment model
    (auth-required, trusted/firewalled LAN) bounds this.

    This is SEPARATE from the ASR download: the audio for ASR is still taken from
    the same ``bestaudio`` stream the legacy path uses (via :func:`transcribe`),
    so a video job and an audio job for one URL yield the same transcript. A
    muxed ``best`` could carry a different audio stream — we never decode ASR
    audio from this video file. Uses a distinct output template + subdir so the
    two downloads cannot overwrite/ambiguate each other's ``info.json``.

    Returns ``(video_path, selected_video_format_id)``.
    """
    frame_dir = work_dir / "frame-stream"
    frame_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(frame_dir / "%(id)s.video.%(ext)s")
    fmt = f"bestvideo[height<={height_cap}]/best[height<={height_cap}]/best"

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", fmt,
        "--no-playlist",
        "-o", out_template,
        "--write-info-json",
        # Only the post-move filepath on stdout (one reliable line); the selected
        # format id is read from the info.json below, not parsed positionally from
        # interleaved --print output.
        "--print", "after_move:filepath",
        "--no-simulate",
        "--",   # end of options: a URL starting with '-' can't be parsed as a flag
        url,
    ]
    log.info("Downloading capped video stream for frames: %s", url)
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("yt-dlp is not available. Install it with `pip install yt-dlp`.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"yt-dlp failed to download video for {url}:\n{exc.stderr}") from exc

    printed = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    path = None
    if printed and os.path.exists(printed[-1]):
        path = Path(printed[-1])
    if path is None:
        # Exclude the `.video.info.json` sidecar (--write-info-json), which yt-dlp
        # writes AFTER the media and would otherwise win the newest-file pick.
        candidates = sorted(
            (p for p in frame_dir.glob("*.video.*") if p.suffix.lower() != ".json"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if candidates:
            path = candidates[0]
    if path is None:
        raise RuntimeError(f"Frame-stream download produced no file in {frame_dir}.")

    # Read the selected format id from the sidecar info.json (reliable, vs parsing
    # interleaved --print stdout). info.json is `<stem-without-.video>.info.json`.
    fmt_id = None
    info_path = path.with_name(path.name.split(".video.")[0] + ".info.json")
    if info_path.is_file():
        import json as _json
        try:
            fmt_id = _json.loads(info_path.read_text(encoding="utf-8")).get("format_id")
        except (OSError, ValueError):
            fmt_id = None
    return path, fmt_id


def ensure_tool(name: str) -> None:
    """Raise a friendly error if a required system binary is missing."""
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required tool '{name}' was not found on your PATH. "
            f"Install it (e.g. `brew install {name}` on macOS, or your OS package manager)."
        )
