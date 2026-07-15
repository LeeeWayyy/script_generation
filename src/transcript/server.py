"""HTTP API server — run this on the GPU host (e.g. the 5090 box).

Accepts a local file upload *or* a URL, runs the transcription pipeline on the
GPU, and serves the result. Jobs run on a single background worker so the GPU is
used serially and the model stays warm in memory across requests.

Run:
    export TRANSCRIPT_TOKEN=$(python -c "import secrets;print(secrets.token_urlsafe(24))")
    transcript-server --host 0.0.0.0 --port 8000

Auth: every request (except /health) must send `Authorization: Bearer <token>`
when TRANSCRIPT_TOKEN is set. If it is unset the server runs OPEN and logs a
loud warning — only acceptable on a trusted, firewalled LAN.
"""

from __future__ import annotations

import argparse
import hmac
import logging
import math
import os
import queue
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# FastAPI types are imported at module level (not lazily) so that the request
# models' stringified annotations — e.g. ``Optional[UploadFile]`` under
# ``from __future__ import annotations`` — resolve against the module namespace.
# server.py is only ever imported on the server host (or in tests), both of which
# have FastAPI; importing ``transcript`` itself stays cheap (it never imports this
# module).
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from . import __version__
from .engine import DEFAULT_MODEL
from .types import is_windows_reserved_basename

log = logging.getLogger("transcript.server")

# Cap an uploaded file so an authenticated client can't fill the server's temp
# disk before the (post-write) archive decompression cap even runs.
MAX_UPLOAD_BYTES = int(os.environ.get("TRANSCRIPT_MAX_UPLOAD_BYTES", 8 * 1024 * 1024 * 1024))
MAX_MULTIPART_OVERHEAD_BYTES = 1024 * 1024
DEFAULT_MAX_QUEUE_SIZE = 32
DEFAULT_JOB_TTL_S = 24 * 3600
DEFAULT_MAX_TERMINAL_JOBS = 100
DEFAULT_JANITOR_INTERVAL_S = 900
_ACTIVE_BUNDLE_TEMPS: set[str] = set()
_ACTIVE_BUNDLE_TEMPS_LOCK = threading.Lock()


class _RequestTooLarge(Exception):
    pass


class BearerAuthMiddleware:
    """Reject unauthenticated requests before anything reads their body."""

    def __init__(self, app, token: Optional[str]):
        self.app = app
        self.expected = f"Bearer {token}".encode() if token else None

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or self.expected is None
                or scope.get("path") == "/health"):
            await self.app(scope, receive, send)
            return
        values = [
            value for name, value in scope.get("headers", ())
            if name.lower() == b"authorization"
        ]
        if len(values) != 1 or not hmac.compare_digest(values[0], self.expected):
            response = JSONResponse(
                {"detail": "Missing or invalid bearer token."}, status_code=401
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


class RequestBodyLimitMiddleware:
    """Reject oversized bodies before or while Starlette parses multipart data."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", ()))
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = 0
        if declared > self.max_bytes:
            await self._reject(scope, receive, send)
            return

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _RequestTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestTooLarge:
            await self._reject(scope, receive, send)

    @staticmethod
    async def _reject(scope, receive, send):
        response = JSONResponse(
            {"detail": "Request body exceeds the size limit."}, status_code=413
        )
        await response(scope, receive, send)


def _save_upload(src, dest: "Path", max_bytes: Optional[int] = None) -> None:
    """Stream an UploadFile to ``dest`` in chunks, aborting (HTTPException 413) if
    it exceeds ``max_bytes`` — never trusts a client-declared length."""
    max_bytes = MAX_UPLOAD_BYTES if max_bytes is None else max_bytes
    written = 0
    try:
        with dest.open("wb") as fh:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(status_code=413, detail="Upload exceeds the size limit.")
                fh.write(chunk)
    except BaseException:
        # Any failure (413, or an OSError like disk-full) must not leave a partial.
        dest.unlink(missing_ok=True)
        raise


def _stage_upload(file) -> "tuple[str, str, str]":
    """Save an upload to a fresh temp dir; return (local_path, source, tmp_dir).
    Cleans up the temp dir on any failure (e.g. a 413) before re-raising."""
    tmp_dir = tempfile.mkdtemp(prefix="transcript-upload-")
    try:
        safe_name = _safe_upload_name(file.filename)
        dest = Path(tmp_dir) / safe_name
        _save_upload(file.file, dest)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)  # don't leak on 413
        raise
    return str(dest), f"upload:{safe_name}", tmp_dir


_STALE_TEMP_AGE_S = 6 * 3600  # only reap temp entries untouched for 6h+


def _sweep_stale_temp_dirs(*, exclude=(), skip_patterns=()) -> None:
    """Reap crash-leaked worker temp entries at startup — the upload/extract/
    enclosure dirs and the ``bundle-*.zip`` files whose ``finally`` /
    BackgroundTask cleanup didn't run on a crash. Only entries untouched for
    ``_STALE_TEMP_AGE_S`` are removed, so a *sibling* server process's currently-
    active temp entries (on a shared host / rolling restart) are left alone."""
    import time as _time
    tmp = Path(tempfile.gettempdir())
    cutoff = _time.time() - _STALE_TEMP_AGE_S
    excluded = {os.path.realpath(str(path)) for path in exclude if path}
    for pattern in ("transcript-upload-*", "transcript-extract-*", "enclosure-*",
                    "bundle-*.zip"):
        if pattern in skip_patterns:
            continue
        for child in tmp.glob(pattern):
            if os.path.realpath(str(child)) in excluded:
                continue
            try:
                if child.stat().st_mtime > cutoff:
                    continue  # recently active — could belong to a live sibling
            except OSError:
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)


def _safe_upload_name(filename: Optional[str]) -> str:
    """Return a safe bare basename for an untrusted multipart upload filename.

    A multipart ``filename`` like ``../victim`` or ``/etc/x`` would otherwise
    escape the temp dir on write — and the worker's cleanup does
    ``rmtree(Path(local_path).parent)``, so a ``../`` name is especially
    dangerous. We strip any directory component and reject the result if it is
    empty or still references the parent.
    """
    if any(ord(char) < 32 or ord(char) == 127 for char in (filename or "")):
        return "upload.bin"
    raw = (filename or "").replace("\\", "/")
    # Reject ":" on the RAW name, before basename. A Windows drive-relative name
    # like "C:foo" or "C:" would otherwise discard the temp dir when joined on a
    # Windows host — and os.path.basename uses ntpath here, which strips the "C:"
    # drive prefix first, so "C:foo" would survive as "foo" if checked post-basename.
    if ":" in raw:
        return "upload.bin"
    base = os.path.basename(raw).strip()
    if not base or base in (".", "..") or "/" in base or "\\" in base:
        return "upload.bin"
    # Reject Windows reserved device names (CON/PRN/AUX/NUL/CLOCK$/COM*/LPT*), which
    # can hang the process or write to a device on a Windows host.
    if is_windows_reserved_basename(base):
        return "upload.bin"  # incl. COM1..COM999 / LPT1..LPT999
    return base


def _validate_url_fields(**values: Optional[str]) -> None:
    """Validate every user-supplied URL before it can reach a fetcher."""
    from .ingest import SsrfError, assert_public_url

    for field_name, value in values.items():
        if value is None:
            continue
        try:
            assert_public_url(value)
        except SsrfError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid {field_name}: {exc}") from exc


def _validate_upload_or_url(url: Optional[str], file: Optional[UploadFile]) -> None:
    if bool(url) == (file is not None):
        raise HTTPException(
            status_code=400, detail="Provide exactly one of 'url' or a 'file' upload."
        )
    if url:
        _validate_url_fields(url=url)


def _validate_speakers(min_speakers: Optional[int], max_speakers: Optional[int],
                       *, diarize: bool) -> None:
    if not diarize and (min_speakers is not None or max_speakers is not None):
        raise HTTPException(
            status_code=400, detail="Speaker hints require diarize=true."
        )
    if min_speakers is not None and min_speakers < 1:
        raise HTTPException(status_code=400, detail="min_speakers must be >= 1.")
    if max_speakers is not None and max_speakers < 1:
        raise HTTPException(status_code=400, detail="max_speakers must be >= 1.")
    if (min_speakers is not None and max_speakers is not None
            and min_speakers > max_speakers):
        raise HTTPException(
            status_code=400, detail="min_speakers must be <= max_speakers."
        )


# ---------------------------------------------------------------------------
# Job model + in-memory store
# ---------------------------------------------------------------------------


@dataclass
class Job:
    id: str
    source: str  # "upload:<name>" or the URL
    status: str = "queued"  # queued | running | done | error
    error: Optional[str] = None
    # Structured failure reason (e.g. a PodcastResolutionError.reason like
    # "ambiguous" vs "stale_selector") so consumers can tell error classes apart.
    error_reason: Optional[str] = None
    # kind=None is the legacy ASR path (Transcript, /jobs routes). A kind in
    # extraction.KINDS routes to the extraction pipeline (ExtractionResult,
    # /extractions routes) — the boundary is structural, never reused.
    kind: Optional[str] = None
    # Pipeline options
    diarize: bool = True
    detect_music: bool = False
    language: Optional[str] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    # Extraction-only options
    frames: bool = False
    cadence_s: Optional[float] = None
    feed_url: Optional[str] = None
    episode_guid: Optional[str] = None
    episode_url: Optional[str] = None
    episode_title: Optional[str] = None
    episode_published: Optional[str] = None
    enclosure_url: Optional[str] = None  # explicit user-supplied podcast enclosure
    # Filled when done (legacy ASR only; extractions persist to the store)
    transcript: object = field(default=None, repr=False)
    # Internal: the local path to process (uploaded temp file or the URL)
    _local_path: Optional[str] = field(default=None, repr=False)
    # Internal: a temp dir WE created for an upload, to remove when the job ends.
    # Explicit ownership beats a startswith(gettempdir()) heuristic, which breaks
    # on macOS where mkdtemp returns /private/var/... but gettempdir() is /var/...
    _upload_tmp_dir: Optional[str] = field(default=None, repr=False)
    _work_tmp_dir: Optional[str] = field(default=None, repr=False)
    created_at: float = field(default_factory=time.time, repr=False)
    finished_at: Optional[float] = field(default=None, repr=False)

    def public(self, queue_position: Optional[int] = None) -> dict:
        d = {
            "id": self.id,
            "source": self.source,
            "status": self.status,
            "diarize": self.diarize,
            "detect_music": self.detect_music,
            "language": self.language,
        }
        if queue_position is not None:
            d["queue_position"] = queue_position
        if self.error:
            d["error"] = self.error
        if self.transcript is not None:
            d["speakers"] = self.transcript.speakers
            d["detected_language"] = self.transcript.language
            d["segments"] = len(self.transcript.segments)
        return d

    def public_extraction(self, queue_position: Optional[int] = None) -> dict:
        """Status view for an extraction job (never touches self.transcript —
        completed extractions are served from the durable ExtractionStore)."""
        d = {
            "id": self.id, "kind": self.kind, "source": self.source,
            "status": self.status, "detect_music": self.detect_music,
        }
        if queue_position is not None:
            d["queue_position"] = queue_position
        if self.error:
            d["error"] = self.error
        if self.error_reason:
            d["error_reason"] = self.error_reason
        return d


class JobStore:
    def __init__(self, *, max_terminal_jobs: int = DEFAULT_MAX_TERMINAL_JOBS,
                 terminal_ttl_s: float = DEFAULT_JOB_TTL_S):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self.max_terminal_jobs = max_terminal_jobs
        self.terminal_ttl_s = terminal_ttl_s

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def claim(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != "queued":
                return None
            job.status = "running"
            return job

    def remove(self, job_id: str, *, statuses: Optional[set[str]] = None) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or (statuses is not None and job.status not in statuses):
                return None
            return self._jobs.pop(job_id)

    def remove_queued(self) -> list[Job]:
        with self._lock:
            queued = [job for job in self._jobs.values() if job.status == "queued"]
            for job in queued:
                self._jobs.pop(job.id, None)
            return queued

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def prune(self, *, now: Optional[float] = None) -> list[str]:
        """Drop expired/overflow terminal jobs, bounding transcript memory."""
        now = time.time() if now is None else now
        with self._lock:
            terminal = [
                job for job in self._jobs.values()
                if job.status in ("done", "error") and job.finished_at is not None
            ]
            expired = {
                job.id for job in terminal
                if now - job.finished_at > self.terminal_ttl_s
            }
            retained = sorted(
                (job for job in terminal if job.id not in expired),
                key=lambda job: job.finished_at,
            )
            overflow = max(0, len(retained) - self.max_terminal_jobs)
            removed = expired | {job.id for job in retained[:overflow]}
            for job_id in removed:
                self._jobs.pop(job_id, None)
            return sorted(removed)


# ---------------------------------------------------------------------------
# Background worker — owns the warm engine, processes one job at a time
# ---------------------------------------------------------------------------


class Worker(threading.Thread):
    _STOP = object()

    def __init__(self, store: JobStore, model: str, device: Optional[str],
                 extraction_store=None, *, max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE):
        super().__init__(daemon=True)
        self.store = store
        self.extraction_store = extraction_store
        self.model = model
        self.device = device
        self.q: "queue.Queue[object]" = queue.Queue(maxsize=max_queue_size)
        self._submit_lock = threading.Lock()
        self._closing = False
        self._engine = None
        self._ocr_engine = None

    def submit(self, job_id: str) -> None:
        with self._submit_lock:
            if self._closing:
                raise queue.Full
            self.q.put_nowait(job_id)

    def cancel(self, job_id: str) -> None:
        """Remove a canceled id from the bounded queue when it is still pending."""
        with self._submit_lock:
            kept = []
            while True:
                try:
                    item = self.q.get_nowait()
                except queue.Empty:
                    break
                self.q.task_done()
                if item != job_id:
                    kept.append(item)
            for item in kept:
                self.q.put_nowait(item)

    def queue_position(self, job_id: str) -> Optional[int]:
        with self.q.mutex:
            pending = [item for item in self.q.queue if item is not self._STOP]
        try:
            return pending.index(job_id) + 1
        except ValueError:
            return None

    def stop(self) -> None:
        with self._submit_lock:
            self._closing = True
            for job in self.store.remove_queued():
                if job._upload_tmp_dir:
                    shutil.rmtree(job._upload_tmp_dir, ignore_errors=True)
            while True:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
                self.q.task_done()
            self.q.put_nowait(self._STOP)

    def _get_engine(self):
        if self._engine is None:
            from .engine import TranscriptionEngine

            log.info("Loading model '%s' (this happens once) ...", self.model)
            self._engine = TranscriptionEngine(model=self.model, device=self.device)
            log.info("Model ready on device=%s.", self._engine.device)
        return self._engine

    def _get_ocr_engine(self):
        if self._ocr_engine is None:
            from .extract import OCR_UNAVAILABLE
            from .ocr import OcrUnavailableError, _load_engine

            log.info("Loading OCR engine (this happens once) ...")
            try:
                self._ocr_engine = _load_engine()
            except OcrUnavailableError as exc:
                log.warning(
                    "OCR unavailable; extraction will continue without text: %s", exc
                )
                self._ocr_engine = OCR_UNAVAILABLE
        return self._ocr_engine

    def run(self) -> None:
        while True:
            try:
                job_id = self.q.get()
                if job_id is self._STOP:
                    return
                job = self.store.claim(job_id)
                if job is None:
                    continue
                log.info("Job %s: running (kind=%s, %s)", job.id, job.kind, job.source)
                try:
                    if job.kind is None:
                        self._run_asr(job)
                    else:
                        self._run_extraction(job)
                except Exception as exc:  # noqa: BLE001 — surface to client
                    job.status = "error"
                    job.error = str(exc)
                    # Preserve a structured reason (e.g. PodcastResolutionError.reason)
                    # so /extractions consumers can distinguish ambiguous vs stale, etc.
                    job.error_reason = getattr(exc, "reason", None)
                    log.exception("Job %s failed", job.id)
                finally:
                    job.finished_at = time.time()
                    # Clean up the upload temp dir WE created (URLs/feeds have none).
                    if job._upload_tmp_dir:
                        shutil.rmtree(job._upload_tmp_dir, ignore_errors=True)
                        job._upload_tmp_dir = None
                    self.store.prune()
            finally:
                self.q.task_done()

    # -- legacy ASR (byte-stable; guarded by the byte-golden test) -----------

    def _run_asr(self, job: Job) -> None:
        from . import transcribe

        engine = self._get_engine()
        work = Path(tempfile.mkdtemp(prefix="transcript-extract-"))
        job._work_tmp_dir = str(work)
        try:
            job.transcript = transcribe(
                job._local_path,
                diarize=job.diarize,
                language=job.language,
                min_speakers=job.min_speakers,
                max_speakers=job.max_speakers,
                engine=engine,
                detect_music=job.detect_music,
                work_dir=str(work),
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)
            job._work_tmp_dir = None
        # Never expose a soon-to-be-deleted server temp path as provenance.
        job.transcript.meta["source"] = job.source
        # Job/server identity → Transcript.meta (the join point: to_json
        # serializes only the Transcript, so Job-level fields must be
        # merged here to reach `-f json`).
        from . import __version__ as _ver
        job.transcript.meta.update({"job_id": job.id, "server_version": _ver})
        job.status = "done"
        log.info("Job %s: done (%d segments)", job.id, len(job.transcript.segments))

    # -- extraction (separate envelope; durable bundle; staging→rename→done) --

    def _run_extraction(self, job: Job) -> None:
        from . import __version__ as _ver
        from . import transcribe
        from .extraction import serialize
        from .extract import (extract_audio_extraction, extract_image_note)

        # audio_extraction is self-contained (it makes its own temp dir); only the
        # asset-producing kinds need a worker work dir.
        work = (Path(tempfile.mkdtemp(prefix="transcript-extract-"))
                if job.kind != "audio_extraction" else None)
        job._work_tmp_dir = str(work) if work is not None else None
        try:
            if job.kind == "image_note":
                result, asset_files = extract_image_note(
                    Path(job._local_path), work, ocr_engine=self._get_ocr_engine()
                )
            elif job.kind == "audio_extraction":
                result, asset_files = extract_audio_extraction(
                    feed_url=job.feed_url, episode_guid=job.episode_guid,
                    episode_url=job.episode_url, episode_title=job.episode_title,
                    episode_published=job.episode_published, enclosure_url=job.enclosure_url,
                    engine=self._get_engine(), transcribe_fn=transcribe,
                    diarize=job.diarize, language=job.language,
                    min_speakers=job.min_speakers, max_speakers=job.max_speakers,
                    detect_music=job.detect_music,
                )
            elif job.kind == "video":
                result, asset_files = self._run_video(job, work, transcribe)
            else:
                raise ValueError(f"unknown extraction kind: {job.kind!r}")

            # Identity → ExtractionResult.meta (NEVER Transcript.meta — leak #2).
            result.meta.update({"job_id": job.id, "server_version": _ver})
            # Publish atomically: build staging bundle, rename into place, THEN done.
            self.extraction_store.record(
                job.id, job.kind, serialize(result), asset_files,
                detect_music=job.detect_music,
            )
            job.status = "done"
            log.info("Job %s: extraction done (%d assets)", job.id, len(asset_files))
        finally:
            if work is not None:
                shutil.rmtree(work, ignore_errors=True)
            job._work_tmp_dir = None

    def _run_video(self, job: Job, work: Path, transcribe):
        """Acquire ASR audio from the SAME bestaudio stream the legacy path uses,
        plus a separate capped video stream for frames (plan §B)."""
        from .extract import extract_video
        from .ingest import download_frame_video, is_url

        # ASR: transcribe() downloads bestaudio/best (URL) or decodes the local
        # media — identical to an audio job for this source.
        transcribe_work = work / "transcribe"
        transcribe_work.mkdir()
        transcript = transcribe(
            job._local_path, diarize=job.diarize, language=job.language,
            min_speakers=job.min_speakers, max_speakers=job.max_speakers,
            engine=self._get_engine(),
            detect_music=job.detect_music,
            work_dir=str(transcribe_work),
        )
        selected_audio_format = transcript.meta.get("selected_format")

        # Frame extraction is the orthogonal opt-in --frames switch. Only download
        # the separate capped video stream / run OCR when frames are requested.
        video_path = None
        selected_video_format = None
        ocr_engine = None
        if job.frames:
            if is_url(job._local_path):
                video_path, selected_video_format = download_frame_video(job._local_path, work)
            else:
                video_path = Path(job._local_path)
            ocr_engine = self._get_ocr_engine()

        return extract_video(
            transcript=transcript, video_path=video_path, asset_dir=work,
            with_frames=job.frames, cadence_s=job.cadence_s, ocr_engine=ocr_engine,
            selected_audio_format=selected_audio_format,
            selected_video_format=selected_video_format,
        )


# ---------------------------------------------------------------------------
# Janitor — TTL eviction of durable extraction bundles (distinct from Worker)
# ---------------------------------------------------------------------------


class Janitor(threading.Thread):
    """Periodic cleanup for terminal jobs, bundles, staging, and stale temp data."""

    def __init__(self, job_store: JobStore, extraction_store, *, interval_s: float = 900.0):
        super().__init__(daemon=True)
        self.job_store = job_store
        self.extraction_store = extraction_store
        self.interval_s = interval_s
        # NB: NOT named `_stop` — that shadows threading.Thread._stop(), which
        # Thread.join() calls internally (→ TypeError: Event not callable).
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def sweep(self) -> None:
        jobs = self.job_store.all()
        running = {j.id for j in jobs if j.status in ("queued", "running")}
        removed = self.job_store.prune()
        if removed:
            log.info("Janitor pruned %d terminal job(s).", len(removed))
        evicted = self.extraction_store.evict_expired(running_ids=running)
        if evicted:
            log.info("Janitor evicted %d expired bundle(s).", len(evicted))
        staged = self.extraction_store.gc_staging(running_ids=running)
        if staged:
            log.info("Janitor reaped %d orphaned staging dir(s).", staged)
        protected = {
            path for job in jobs for path in (job._upload_tmp_dir, job._work_tmp_dir)
            if path
        }
        with _ACTIVE_BUNDLE_TEMPS_LOCK:
            protected.update(_ACTIVE_BUNDLE_TEMPS)
        skip = {"enclosure-*"} if any(
            job.kind == "audio_extraction" and job.status == "running" for job in jobs
        ) else set()
        _sweep_stale_temp_dirs(exclude=protected, skip_patterns=skip)

    def run(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            try:
                self.sweep()
            except Exception:  # noqa: BLE001 — never let the sweeper die
                log.exception("Janitor sweep failed")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


def create_app(model: str = DEFAULT_MODEL, device: Optional[str] = None):
    from .formats import FORMATS, render

    token = os.environ.get("TRANSCRIPT_TOKEN")
    if not token:
        log.warning(
            "TRANSCRIPT_TOKEN is not set — the server is running WITHOUT authentication. "
            "Only do this on a trusted, firewalled network."
        )

    from .extraction import KINDS as EXTRACTION_KINDS
    from .extraction_store import ExtractionStore

    max_queue_size = int(os.environ.get("TRANSCRIPT_MAX_QUEUE_SIZE", DEFAULT_MAX_QUEUE_SIZE))
    job_ttl_s = float(os.environ.get("TRANSCRIPT_JOB_TTL_SECONDS", DEFAULT_JOB_TTL_S))
    max_terminal_jobs = int(os.environ.get(
        "TRANSCRIPT_MAX_TERMINAL_JOBS", DEFAULT_MAX_TERMINAL_JOBS
    ))
    janitor_interval_s = float(os.environ.get(
        "TRANSCRIPT_JANITOR_INTERVAL_SECONDS", DEFAULT_JANITOR_INTERVAL_S
    ))
    max_concurrent_bundles = int(os.environ.get("TRANSCRIPT_MAX_CONCURRENT_BUNDLES", 8))
    if (MAX_UPLOAD_BYTES <= 0 or max_queue_size <= 0 or max_terminal_jobs <= 0
            or max_concurrent_bundles <= 0
            or not math.isfinite(job_ttl_s) or job_ttl_s <= 0
            or not math.isfinite(janitor_interval_s) or janitor_interval_s <= 0):
        raise ValueError("server queue/retention/janitor limits must be positive")

    store = JobStore(
        max_terminal_jobs=max_terminal_jobs, terminal_ttl_s=job_ttl_s,
    )
    extraction_store = ExtractionStore()  # scans existing bundles on construction
    worker = Worker(
        store, model=model, device=device, extraction_store=extraction_store,
        max_queue_size=max_queue_size,
    )
    janitor = Janitor(store, extraction_store, interval_s=janitor_interval_s)
    # Bound concurrent /bundle builds so simultaneous downloads can't exhaust the
    # server's temp disk with full-zip copies.
    bundle_sem = threading.Semaphore(max_concurrent_bundles)

    @asynccontextmanager
    async def lifespan(_app):
        _sweep_stale_temp_dirs()
        worker.start()
        janitor.start()
        try:
            yield
        finally:
            janitor.stop()
            worker.stop()
            janitor.join()
            worker.join()

    app = FastAPI(title="transcript", version=__version__, lifespan=lifespan)
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_bytes=MAX_UPLOAD_BYTES + MAX_MULTIPART_OVERHEAD_BYTES,
    )
    # Added last so Starlette places it outside the body limiter/parser stack.
    app.add_middleware(BearerAuthMiddleware, token=token)
    app.state.job_store = store
    app.state.worker = worker
    app.state.extraction_store = extraction_store

    def auth(authorization: str | None = Header(default=None)) -> None:
        if not token:
            return
        expected = f"Bearer {token}"
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Missing or invalid bearer token.")

    def _public(job: Job) -> dict:
        position = worker.queue_position(job.id) if job.status == "queued" else None
        return job.public_extraction(position) if job.kind else job.public(position)

    def _enqueue(job: Job) -> dict:
        store.add(job)
        try:
            worker.submit(job.id)
        except queue.Full as exc:
            store.remove(job.id, statuses={"queued"})
            if job._upload_tmp_dir:
                shutil.rmtree(job._upload_tmp_dir, ignore_errors=True)
            raise HTTPException(
                status_code=503, detail="Job queue is full; retry shortly."
            ) from exc
        return _public(job)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model, "queued_or_running": [
            j.id for j in store.all() if j.status in ("queued", "running")
        ]}

    @app.post("/jobs")
    def create_job(
        _: None = Depends(auth),
        url: Optional[str] = Form(default=None),
        file: Optional[UploadFile] = File(default=None),
        diarize: bool = Form(default=True),
        detect_music: bool = Form(default=False),
        language: Optional[str] = Form(default=None),
        min_speakers: Optional[int] = Form(default=None),
        max_speakers: Optional[int] = Form(default=None),
    ):
        _validate_upload_or_url(url, file)
        _validate_speakers(min_speakers, max_speakers, diarize=diarize)

        job_id = uuid.uuid4().hex[:12]

        upload_tmp_dir = None
        if file is not None:
            local_path, source, upload_tmp_dir = _stage_upload(file)
        else:
            local_path, source = url, url

        job = Job(
            id=job_id,
            source=source,
            diarize=diarize,
            detect_music=detect_music,
            language=language,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        job._local_path = local_path
        job._upload_tmp_dir = upload_tmp_dir
        return _enqueue(job)

    # The legacy /jobs API is Transcript-ONLY: extraction jobs share the in-memory
    # JobStore but must never be visible or fetchable here (else /jobs/{id}/result
    # would call render() on a None transcript → 500). They live behind /extractions.
    def _asr_job(job_id: str) -> Optional[Job]:
        job = store.get(job_id)
        return job if (job is not None and job.kind is None) else None

    @app.get("/jobs")
    def list_jobs(_: None = Depends(auth)):
        return [_public(j) for j in store.all() if j.kind is None]

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str, _: None = Depends(auth)):
        job = _asr_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job.")
        return _public(job)

    @app.get("/jobs/{job_id}/result", response_class=PlainTextResponse)
    def get_result(job_id: str, format: str = "txt", _: None = Depends(auth)):
        job = _asr_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job.")
        if job.status == "error":
            raise HTTPException(status_code=409, detail=f"Job failed: {job.error}")
        if job.status != "done":
            raise HTTPException(status_code=409, detail=f"Job not finished (status: {job.status}).")
        if format not in FORMATS:
            raise HTTPException(status_code=400, detail=f"format must be one of {FORMATS}")
        return render(job.transcript, format)

    @app.delete("/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str, _: None = Depends(auth)):
        if _asr_job(job_id) is None:
            raise HTTPException(status_code=404, detail="No such job.")
        removed = store.remove(job_id, statuses={"queued", "done", "error"})
        if removed is None:
            if _asr_job(job_id) is None:
                raise HTTPException(status_code=404, detail="No such job.")
            raise HTTPException(status_code=409, detail="A running job cannot be deleted.")
        if removed.status == "queued":
            worker.cancel(job_id)
            if removed._upload_tmp_dir:
                shutil.rmtree(removed._upload_tmp_dir, ignore_errors=True)
        return Response(status_code=204)

    # -- extraction routes (separate envelope; auth required; never the /jobs API) --

    @app.post("/extractions")
    def create_extraction(
        _: None = Depends(auth),
        kind: str = Form(...),
        url: Optional[str] = Form(default=None),
        file: Optional[UploadFile] = File(default=None),
        diarize: bool = Form(default=True),
        detect_music: bool = Form(default=False),
        language: Optional[str] = Form(default=None),
        min_speakers: Optional[int] = Form(default=None),
        max_speakers: Optional[int] = Form(default=None),
        frames: bool = Form(default=False),
        cadence_s: Optional[float] = Form(default=None),
        feed_url: Optional[str] = Form(default=None),
        episode_guid: Optional[str] = Form(default=None),
        episode_url: Optional[str] = Form(default=None),
        episode_title: Optional[str] = Form(default=None),
        episode_published: Optional[str] = Form(default=None),
        enclosure_url: Optional[str] = Form(default=None),
    ):
        if kind not in EXTRACTION_KINDS:
            raise HTTPException(status_code=400, detail=f"kind must be one of {EXTRACTION_KINDS}")
        _validate_speakers(min_speakers, max_speakers, diarize=diarize)
        selectors = (episode_guid, episode_url, episode_title, episode_published)

        if kind == "image_note":
            if file is None:
                raise HTTPException(
                    status_code=400,
                    detail="kind=image_note requires a 'file' upload (a zip/tar archive).",
                )
            if url or feed_url or enclosure_url or any(selectors):
                raise HTTPException(
                    status_code=400, detail="kind=image_note accepts only a file upload."
                )
            if (not diarize or detect_music
                    or min_speakers is not None or max_speakers is not None
                    or frames or cadence_s is not None or language is not None):
                raise HTTPException(
                    status_code=400, detail="Option is not applicable to kind=image_note."
                )
        elif kind == "video":
            _validate_upload_or_url(url, file)
            if feed_url or enclosure_url or any(selectors):
                raise HTTPException(
                    status_code=400, detail="kind=video does not accept podcast fields."
                )
            if cadence_s is not None and not frames:
                raise HTTPException(
                    status_code=400, detail="cadence_s requires frames=true."
                )
        else:
            if url:
                raise HTTPException(
                    status_code=400, detail="kind=audio_extraction does not accept 'url'."
                )
            if file is not None:
                raise HTTPException(
                    status_code=400,
                    detail="kind=audio_extraction does not accept a file upload.",
                )
            if bool(feed_url) == bool(enclosure_url):
                raise HTTPException(
                    status_code=400,
                    detail="Provide exactly one of 'feed_url' or 'enclosure_url'.",
                )
            title_pair = bool(episode_title) and bool(episode_published)
            if bool(episode_title) != bool(episode_published):
                raise HTTPException(
                    status_code=400,
                    detail="episode_title and episode_published must be provided together.",
                )
            if title_pair and (episode_guid or episode_url):
                raise HTTPException(
                    status_code=400,
                    detail="The title/published selector cannot accompany GUID or episode_url.",
                )
            if feed_url and not (episode_guid or episode_url or title_pair):
                raise HTTPException(
                    status_code=400,
                    detail="feed_url requires episode_guid, episode_url, or the full "
                    "episode_title/episode_published pair.",
                )
            if enclosure_url and any(selectors):
                raise HTTPException(
                    status_code=400,
                    detail="Podcast episode selectors cannot accompany enclosure_url.",
                )
            if frames or cadence_s is not None:
                raise HTTPException(
                    status_code=400,
                    detail="frames and cadence_s are not applicable to audio_extraction.",
                )
            _validate_url_fields(
                feed_url=feed_url, enclosure_url=enclosure_url, episode_url=episode_url,
            )
        if cadence_s is not None:
            from .frames import MIN_CADENCE_S
            if not math.isfinite(cadence_s) or cadence_s < MIN_CADENCE_S:
                raise HTTPException(
                    status_code=400,
                    detail=f"cadence_s must be >= {MIN_CADENCE_S} seconds.",
                )

        job_id = uuid.uuid4().hex[:12]
        upload_tmp_dir = None
        if file is not None:
            local_path, source, upload_tmp_dir = _stage_upload(file)
        else:
            local_path = source = url or feed_url or enclosure_url

        job = Job(
            id=job_id, source=source, kind=kind, diarize=diarize,
            detect_music=detect_music, language=language,
            min_speakers=min_speakers, max_speakers=max_speakers,
            frames=frames, cadence_s=cadence_s, feed_url=feed_url,
            episode_guid=episode_guid, episode_url=episode_url,
            episode_title=episode_title, episode_published=episode_published,
            enclosure_url=enclosure_url,
        )
        job._local_path = local_path
        job._upload_tmp_dir = upload_tmp_dir
        return _enqueue(job)

    import re as _re
    _JOB_ID_RE = _re.compile(r"^[0-9a-f]{12}$")

    def _valid_job_id(job_id: str) -> None:
        # job ids are uuid4().hex[:12]; reject anything else structurally so a path
        # component can never reach the store's `root / job_id` join.
        if not _JOB_ID_RE.match(job_id):
            raise HTTPException(status_code=404, detail="No such extraction.")

    @app.get("/extractions/{job_id}")
    def get_extraction(job_id: str, _: None = Depends(auth)):
        _valid_job_id(job_id)
        rec = extraction_store.get(job_id)
        job = store.get(job_id)
        # rec present but the on-disk bundle vanished (deleted out-of-band, or a
        # stale index entry) → 410, consistent with /result and /bundle.
        if rec is not None and extraction_store.result_path(job_id).is_file():
            if job is not None and job.kind is not None:
                return _public(job)
            return {
                "id": job_id, "kind": rec.get("kind"), "status": "done",
                "detect_music": rec.get("detect_music", False),
            }
        if rec is not None:
            raise HTTPException(status_code=410, detail="Bundle was evicted or lost.")
        if job is not None and job.kind is not None:
            # In-memory says done but the durable bundle is gone → evicted/lost.
            if job.status == "done":
                raise HTTPException(status_code=410, detail="Bundle was evicted or lost.")
            return _public(job)
        if extraction_store.was_evicted(job_id):  # known-but-evicted (no in-memory job)
            raise HTTPException(status_code=410, detail="Bundle was evicted or lost.")
        raise HTTPException(status_code=404, detail="No such extraction.")

    @app.get("/extractions/{job_id}/result", response_class=PlainTextResponse)
    def get_extraction_result(job_id: str, _: None = Depends(auth)):
        _valid_job_id(job_id)
        # Durable, completed result (also bumps last-access — /result shares the
        # bundle read-lease so a client doesn't lose /bundle to TTL between calls).
        text = extraction_store.read_result(job_id, bump=True)
        if text is not None:
            return PlainTextResponse(text, media_type="application/json")
        # The durable index may still know this id even though result.json is gone
        # (out-of-band loss / a race) → that is "known but gone" = 410, same as the
        # status route, not 404.
        if extraction_store.get(job_id) is not None or extraction_store.was_evicted(job_id):
            raise HTTPException(status_code=410, detail="Bundle was evicted or lost.")
        job = store.get(job_id)
        if job is None or job.kind is None:
            raise HTTPException(status_code=404, detail="No such extraction.")
        if job.status == "error":
            reason = f" [{job.error_reason}]" if job.error_reason else ""
            raise HTTPException(status_code=409, detail=f"Extraction failed{reason}: {job.error}")
        if job.status != "done":
            raise HTTPException(
                status_code=409, detail=f"Extraction not finished (status: {job.status})."
            )
        # In-memory says done but the durable bundle is gone → evicted/lost.
        raise HTTPException(status_code=410, detail="Bundle was evicted or lost on restart.")

    @app.delete("/extractions/{job_id}", status_code=204)
    def delete_extraction(job_id: str, _: None = Depends(auth)):
        _valid_job_id(job_id)
        job = store.get(job_id)
        if job is not None and job.kind is None:
            raise HTTPException(status_code=404, detail="No such extraction.")
        if job is not None and job.status == "queued":
            removed = store.remove(job_id, statuses={"queued"})
            if removed is not None:
                worker.cancel(job_id)
                if removed._upload_tmp_dir:
                    shutil.rmtree(removed._upload_tmp_dir, ignore_errors=True)
                return Response(status_code=204)
            job = store.get(job_id)
        if job is not None and job.status == "running":
            raise HTTPException(status_code=409, detail="A running extraction cannot be deleted.")
        if job is not None and job.status == "error":
            store.remove(job_id, statuses={"error"})
            return Response(status_code=204)

        deleted = extraction_store.delete(job_id)
        if deleted == "leased":
            raise HTTPException(status_code=409, detail="Extraction bundle is in use.")
        if deleted == "deleted":
            store.remove(job_id, statuses={"done"})
            return Response(status_code=204)
        if extraction_store.was_evicted(job_id) or (job is not None and job.status == "done"):
            raise HTTPException(status_code=410, detail="Bundle was evicted or lost.")
        raise HTTPException(status_code=404, detail="No such extraction.")

    @app.get("/extractions/{job_id}/bundle")
    def get_extraction_bundle(job_id: str, _: None = Depends(auth)):
        _valid_job_id(job_id)
        import os as _os
        import tempfile
        import zipfile

        from fastapi.responses import StreamingResponse
        from starlette.background import BackgroundTask

        # Build the zip to a TEMP FILE under the lease (so the asset files can't be
        # evicted mid-build), then stream that file back. We never hold the whole
        # bundle — which can approach the 2 GiB archive cap — in RAM. Zip
        # determinism is NOT required: the consumer hashes `text` + per-asset
        # sha256, not the zip.
        # Bound concurrent builds so simultaneous downloads can't fill temp disk.
        if not bundle_sem.acquire(blocking=False):
            raise HTTPException(status_code=503,
                                detail="Too many concurrent bundle downloads; retry shortly.")
        try:
            with extraction_store.lease(job_id) as job_dir:
                if job_dir is None:
                    # known-but-gone (in-memory done OR the durable index has it)
                    # → 410, consistent with /result and the status route.
                    job = store.get(job_id)
                    known = ((job is not None and job.kind is not None and job.status == "done")
                             or extraction_store.get(job_id) is not None
                             or extraction_store.was_evicted(job_id))
                    if known:
                        raise HTTPException(status_code=410, detail="Bundle was evicted or lost.")
                    raise HTTPException(status_code=404, detail="No such extraction bundle.")
                tmp = tempfile.NamedTemporaryFile(prefix="bundle-", suffix=".zip", delete=False)
                try:
                    tmp.close()
                    with _ACTIVE_BUNDLE_TEMPS_LOCK:
                        _ACTIVE_BUNDLE_TEMPS.add(tmp.name)
                    manifest_path = job_dir / ExtractionStore.MANIFEST_NAME
                    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
                        for path in sorted(job_dir.rglob("*")):
                            # Exclude the mutable side manifest by EXACT path (not
                            # name) so a stray nested manifest.json isn't dropped.
                            if path.is_file() and path != manifest_path:
                                compressed_image = path.suffix.lower() in {
                                    ".avif", ".gif", ".heic", ".heif", ".jpeg", ".jpg",
                                    ".png", ".webp",
                                }
                                zf.write(
                                    path, arcname=path.relative_to(job_dir).as_posix(),
                                    compress_type=(zipfile.ZIP_STORED if compressed_image
                                                   else zipfile.ZIP_DEFLATED),
                                )
                except BaseException:
                    with _ACTIVE_BUNDLE_TEMPS_LOCK:
                        _ACTIVE_BUNDLE_TEMPS.discard(tmp.name)
                    _os.unlink(tmp.name)  # don't leak the temp file if the build fails
                    raise
        except BaseException:
            bundle_sem.release()
            raise

        def _stream():
            with open(tmp.name, "rb") as fh:
                while True:
                    chunk = fh.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk

        def _cleanup():
            with _ACTIVE_BUNDLE_TEMPS_LOCK:
                _ACTIVE_BUNDLE_TEMPS.discard(tmp.name)
            try:
                _os.unlink(tmp.name)
            except OSError:
                pass
            bundle_sem.release()  # held across the whole stream; released when done

        # A BackgroundTask runs after the response finishes (incl. a dropped
        # connection) without relying on generator-finalizer GC timing.
        try:
            size = _os.path.getsize(tmp.name)
        except OSError:
            _cleanup()
            raise
        return StreamingResponse(_stream(), media_type="application/zip",
                                 headers={"Content-Length": str(size)},
                                 background=BackgroundTask(_cleanup))

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="transcript-server", description="Transcription HTTP API.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Whisper model (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Force device (default: auto).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import uvicorn

    app = create_app(model=args.model, device=args.device)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
