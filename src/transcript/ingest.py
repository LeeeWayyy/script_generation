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
import json
from pathlib import Path

from .types import Segment

log = logging.getLogger("transcript.ingest")


def is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def is_youtube_url(source: str) -> bool:
    if not is_url(source):
        return False
    from urllib.parse import urlsplit

    host = (urlsplit(source).hostname or "").lower()
    return host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com")


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


def _ytdlp_fetch(url: str, *, fmt: str, out_template: str, search_dir: Path,
                 fallback_glob: str, fail_action: str, nofile_msg: str,
                 extra_args: tuple[str, ...] = ()) -> Path:
    """Run yt-dlp for ``url`` and return the produced file path.

    Shared by the audio (`_download_url`) and frame-video (`download_frame_video`)
    paths: identical invocation (best stream of ``fmt`` → ``out_template`` with
    ``--write-info-json``), the same ``after_move:filepath`` stdout parse, and the
    same newest-file fallback that EXCLUDES the ``.info.json`` sidecar (written
    AFTER the media, so a naive newest-file pick would wrongly return the JSON).
    ``fail_action`` (e.g. "download") and ``nofile_msg`` only shape the two
    error messages, kept verbatim per caller.
    """
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", fmt,
        "--no-playlist",
        "-o", out_template,
        "--write-info-json",   # full provenance metadata → <id>.info.json
        "--print", "after_move:filepath",
        "--no-simulate",
        *extra_args,
        "--",   # end of options: a URL starting with '-' can't be parsed as a flag
        url,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "yt-dlp is not available. Install it with `pip install yt-dlp`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"yt-dlp failed to {fail_action} {url}:\n{exc.stderr}") from exc

    # The last non-empty stdout line is the final file path (after_move:filepath).
    printed = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    if printed and os.path.exists(printed[-1]):
        return Path(printed[-1])
    candidates = sorted(
        (p for p in search_dir.glob(fallback_glob)
         if p.suffix.lower() not in {".json", ".json3"}),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if candidates:
        return candidates[0]
    raise RuntimeError(nofile_msg)


def _download_url(url: str, work_dir: Path, *, subtitle_language: str | None = None) -> Path:
    """Download best-audio for ``url`` using yt-dlp; return the downloaded file path."""
    work_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s with yt-dlp ...", url)
    # %(id)s keeps the name predictable and filesystem-safe. The info.json sidecar
    # is read back by transcribe() for provenance.
    subtitle_args = (() if subtitle_language is None else (
        "--write-subs", "--no-write-auto-subs", "--sub-langs", subtitle_language,
        "--sub-format", "json3",
    ))
    return _ytdlp_fetch(
        url, fmt="bestaudio/best", out_template=str(work_dir / "%(id)s.%(ext)s"),
        search_dir=work_dir, fallback_glob="*", fail_action="download",
        nofile_msg=f"Download appeared to succeed but no file was found in {work_dir}.",
        extra_args=subtitle_args,
    )


def _manual_caption_language(info: dict, preferred: str | None) -> str | None:
    """Choose a creator-supplied JSON3 subtitle track; auto captions are separate."""
    tracks = {
        lang: formats for lang, formats in (info.get("subtitles") or {}).items()
        if lang != "live_chat" and any(f.get("ext") == "json3" for f in formats)
    }
    if not tracks:
        return None

    def match(wanted: str | None) -> str | None:
        if not wanted:
            return None
        wanted = wanted.lower()
        return next((lang for lang in tracks if lang.lower() == wanted), None) or next(
            (lang for lang in tracks
             if lang.lower().split("-", 1)[0] == wanted.split("-", 1)[0]), None
        )

    if preferred:
        return match(preferred)
    return match(info.get("language")) or match(info.get("original_language")) or next(iter(tracks))


def download_manual_caption(
    url: str, work_dir: Path, *, language: str | None = None, with_audio: bool = False,
) -> tuple[Path | None, str, Path | None, dict] | None:
    """Download a human YouTube caption, optionally with the audio needed for diarization.

    Returns ``(caption_path, caption_language, media_path, info)``. Absence means
    there is no matching manual track; ``automatic_captions`` is never inspected.
    """
    probe = [
        sys.executable, "-m", "yt_dlp", "--dump-single-json", "--skip-download",
        "--no-playlist", "--no-warnings", "--", url,
    ]
    try:
        info = json.loads(subprocess.run(
            probe, check=True, capture_output=True, text=True,
        ).stdout)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        log.warning("Could not inspect YouTube captions (%s); falling back to ASR.", exc)
        return None

    caption_language = _manual_caption_language(info, language)
    if caption_language is None:
        return None

    work_dir.mkdir(parents=True, exist_ok=True)
    media = None
    if with_audio:
        media = _download_url(url, work_dir, subtitle_language=caption_language)
    else:
        cmd = [
            sys.executable, "-m", "yt_dlp", "--skip-download", "--write-subs",
            "--no-write-auto-subs", "--sub-langs", caption_language,
            "--sub-format", "json3", "--write-info-json", "--no-playlist",
            "-o", str(work_dir / "%(id)s.%(ext)s"), "--", url,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            log.warning("Could not download manual captions (%s); falling back to ASR.", exc)
            return None, caption_language, None, info

    video_id = str(info.get("id") or "")
    suffix = f".{caption_language}.json3"
    candidates = [
        p for p in work_dir.glob("*.json3")
        if p.name.endswith(suffix) and (not video_id or p.name.startswith(f"{video_id}."))
    ]
    caption = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
    return caption, caption_language, media, info


def parse_json3_caption(path: Path) -> list[Segment]:
    """Parse the timing/text subset of YouTube's JSON3 subtitle format."""
    from html import unescape

    data = json.loads(path.read_text(encoding="utf-8"))
    segments: list[Segment] = []
    for event in data.get("events", []):
        text = unescape("".join(part.get("utf8", "") for part in event.get("segs", [])))
        text = " ".join(text.split())
        if not text:
            continue
        start = event.get("tStartMs", 0) / 1000
        duration = event.get("dDurationMs", 0) / 1000
        segments.append(Segment(text=text, start=start, end=start + duration))
    return segments


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
    fmt = f"bestvideo[height<={height_cap}]/best[height<={height_cap}]/best"
    log.info("Downloading capped video stream for frames: %s", url)
    # Distinct output template + subdir + glob so this download can't overwrite /
    # ambiguate the ASR `bestaudio` download's info.json. The selected format id is
    # read from the info.json below, never parsed positionally from --print output.
    path = _ytdlp_fetch(
        url, fmt=fmt, out_template=str(frame_dir / "%(id)s.video.%(ext)s"),
        search_dir=frame_dir, fallback_glob="*.video.*", fail_action="download video for",
        nofile_msg=f"Frame-stream download produced no file in {frame_dir}.",
    )

    # Read the selected format id from the sidecar info.json (reliable, vs parsing
    # interleaved --print stdout). For media "<id>.video.<ext>", --write-info-json
    # writes "<id>.video.info.json" (the media ext swapped for ".info.json").
    fmt_id = None
    info_path = path.with_suffix(".info.json")
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
