"""Turn a *source* (local path or URL) into a local media file on disk.

- Local path: validated and returned as-is.
- URL: downloaded with yt-dlp (best available audio) into ``work_dir``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .types import Segment

log = logging.getLogger("transcript.ingest")

DEFAULT_MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024 * 1024  # match the default upload cap
DEFAULT_DOWNLOAD_TIMEOUT_S = 3600.0
PROCESS_STOP_TIMEOUT_S = 1.0


def _download_limits() -> tuple[int, float]:
    """Read operator-configured yt-dlp limits at fetch time."""
    try:
        max_bytes = int(os.environ.get(
            "TRANSCRIPT_MAX_DOWNLOAD_BYTES", DEFAULT_MAX_DOWNLOAD_BYTES,
        ))
        timeout = float(os.environ.get(
            "TRANSCRIPT_DOWNLOAD_TIMEOUT_S", DEFAULT_DOWNLOAD_TIMEOUT_S,
        ))
    except ValueError as exc:
        raise RuntimeError("yt-dlp download limits must be numeric") from exc
    if max_bytes <= 0 or not math.isfinite(timeout) or timeout <= 0:
        raise RuntimeError("yt-dlp download limits must be finite and greater than zero")
    return max_bytes, timeout


def is_url(source: str) -> bool:
    from urllib.parse import urlsplit

    try:
        return urlsplit(source).scheme.lower() in ("http", "https")
    except ValueError:
        return False


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

    try:
        parts = urlsplit(url)
        port = parts.port or (443 if parts.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise SsrfError(f"invalid URL: {url!r}") from exc
    if parts.scheme.lower() not in ("http", "https"):
        raise SsrfError(f"URL scheme not allowed: {url!r}")
    host = parts.hostname
    if not host:
        raise SsrfError(f"no host in URL: {url!r}")
    if os.environ.get("TRANSCRIPT_ALLOW_PRIVATE_FETCH") == "1":
        return
    try:
        infos = socket.getaddrinfo(host, port)
    except socket.gaierror as exc:
        raise SsrfError(f"cannot resolve host {host!r}: {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or getattr(ip, "is_site_local", False):
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
    max_bytes, timeout = _download_limits()
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--ignore-config",
        "-f", fmt,
        "--max-filesize", str(max_bytes),
        "--no-playlist",
        "-o", out_template,
        "--write-info-json",   # full provenance metadata → <id>.info.json
        "--print", "after_move:filepath",
        "--no-simulate",
        *extra_args,
        "--",   # end of options: a URL starting with '-' can't be parsed as a flag
        url,
    ]

    def tracked_files() -> dict[Path, int]:
        tracked = {}
        for path in search_dir.glob(fallback_glob):
            try:
                if path.is_file():
                    tracked[path] = path.stat().st_size
            except OSError:
                pass
        return tracked

    before = tracked_files()

    def downloaded_bytes() -> int:
        return sum(
            max(0, size - before.get(path, 0))
            for path, size in tracked_files().items()
        )

    def cleanup_download() -> None:
        for path, size in tracked_files().items():
            if path not in before or size != before[path]:
                try:
                    path.unlink()
                except OSError:
                    pass

    popen_kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    elif os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "yt-dlp is not available. Install it with `pip install yt-dlp`."
        ) from exc

    def stop_process() -> None:
        pid = getattr(proc, "pid", None)
        if os.name == "posix" and pid is not None:
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass
        elif os.name == "nt" and pid is not None:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=PROCESS_STOP_TIMEOUT_S, check=False,
                )
            except (OSError, subprocess.SubprocessError):
                pass
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.communicate(timeout=PROCESS_STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            # A surviving descendant may still own the inherited pipe handles.
            for pipe in (getattr(proc, "stdout", None), getattr(proc, "stderr", None)):
                if pipe is not None:
                    pipe.close()

    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            stop_process()
            cleanup_download()
            raise RuntimeError(
                f"yt-dlp timed out after {timeout:g}s while trying to {fail_action} {url}"
            )
        try:
            stdout, stderr = proc.communicate(timeout=min(0.1, remaining))
            break
        except subprocess.TimeoutExpired as exc:
            if downloaded_bytes() <= max_bytes:
                continue
            stop_process()
            cleanup_download()
            raise RuntimeError(
                f"yt-dlp download exceeds the {max_bytes}-byte cap: {url}"
            ) from exc
        except BaseException:
            stop_process()
            cleanup_download()
            raise

    if downloaded_bytes() > max_bytes:
        cleanup_download()
        raise RuntimeError(f"yt-dlp download exceeds the {max_bytes}-byte cap: {url}")
    if proc.returncode:
        if "max-filesize" in stderr.lower() or "larger than" in stderr.lower():
            cleanup_download()
            raise RuntimeError(
                f"yt-dlp refused {url}: download exceeds the {max_bytes}-byte cap"
            )
        cleanup_download()
        raise RuntimeError(f"yt-dlp failed to {fail_action} {url}:\n{stderr}")

    def verify(path: Path) -> Path:
        if path.stat().st_size > max_bytes:
            cleanup_download()
            path.unlink(missing_ok=True)
            raise RuntimeError(
                f"yt-dlp download exceeds the {max_bytes}-byte cap: {url}"
            )
        return path

    # The last non-empty stdout line is the final file path (after_move:filepath).
    printed = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    if printed and os.path.exists(printed[-1]):
        return verify(Path(printed[-1]))
    candidates = sorted(
        (p for p in search_dir.glob(fallback_glob)
         if p.suffix.lower() not in {".json", ".json3"}),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if candidates:
        return verify(candidates[0])
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
        sys.executable, "-m", "yt_dlp", "--ignore-config",
        "--dump-single-json", "--skip-download",
        "--no-playlist", "--no-warnings", "--", url,
    ]
    try:
        info = json.loads(subprocess.run(
            probe, check=True, capture_output=True, text=True,
            timeout=_download_limits()[1],
        ).stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
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
            sys.executable, "-m", "yt_dlp", "--ignore-config",
            "--skip-download", "--write-subs",
            "--no-write-auto-subs", "--sub-langs", caption_language,
            "--sub-format", "json3", "--write-info-json", "--no-playlist",
            "-o", str(work_dir / "%(id)s.%(ext)s"), "--", url,
        ]
        try:
            subprocess.run(
                cmd, check=True, capture_output=True, text=True,
                timeout=_download_limits()[1],
            )
        except (OSError, subprocess.SubprocessError) as exc:
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
    fmt = f"bestvideo[height<={height_cap}]/best[height<={height_cap}]"
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
