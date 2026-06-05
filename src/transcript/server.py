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
import logging
import os
import queue
import shutil
import tempfile
import threading
import uuid
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
from fastapi.responses import PlainTextResponse

log = logging.getLogger("transcript.server")


def _safe_upload_name(filename: Optional[str]) -> str:
    """Return a safe bare basename for an untrusted multipart upload filename.

    A multipart ``filename`` like ``../victim`` or ``/etc/x`` would otherwise
    escape the temp dir on write — and the worker's cleanup does
    ``rmtree(Path(local_path).parent)``, so a ``../`` name is especially
    dangerous. We strip any directory component and reject the result if it is
    empty or still references the parent.
    """
    base = os.path.basename((filename or "").replace("\\", "/")).strip()
    if not base or base in (".", "..") or "/" in base or "\\" in base:
        return "upload.bin"
    # Reject Windows reserved device names (CON/PRN/AUX/NUL/COM1-9/LPT1-9), which
    # can hang the process or write to a device on a Windows host.
    stem = os.path.splitext(base)[0].upper()
    if stem in {"CON", "PRN", "AUX", "NUL"} or (
        len(stem) == 4 and stem[:3] in {"COM", "LPT"} and stem[3].isdigit()
    ):
        return "upload.bin"
    return base


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

    def public(self) -> dict:
        d = {
            "id": self.id,
            "source": self.source,
            "status": self.status,
            "diarize": self.diarize,
            "language": self.language,
        }
        if self.error:
            d["error"] = self.error
        if self.transcript is not None:
            d["speakers"] = self.transcript.speakers
            d["detected_language"] = self.transcript.language
            d["segments"] = len(self.transcript.segments)
        return d

    def public_extraction(self) -> dict:
        """Status view for an extraction job (never touches self.transcript —
        completed extractions are served from the durable ExtractionStore)."""
        d = {"id": self.id, "kind": self.kind, "source": self.source, "status": self.status}
        if self.error:
            d["error"] = self.error
        if self.error_reason:
            d["error_reason"] = self.error_reason
        return d


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())


# ---------------------------------------------------------------------------
# Background worker — owns the warm engine, processes one job at a time
# ---------------------------------------------------------------------------


class Worker(threading.Thread):
    def __init__(self, store: JobStore, model: str, device: Optional[str],
                 extraction_store=None):
        super().__init__(daemon=True)
        self.store = store
        self.extraction_store = extraction_store
        self.model = model
        self.device = device
        self.q: "queue.Queue[str]" = queue.Queue()
        self._engine = None
        self._ocr_engine = None

    def submit(self, job_id: str) -> None:
        self.q.put(job_id)

    def _get_engine(self):
        if self._engine is None:
            from .engine import TranscriptionEngine

            log.info("Loading model '%s' (this happens once) ...", self.model)
            self._engine = TranscriptionEngine(model=self.model, device=self.device)
            log.info("Model ready on device=%s.", self._engine.device)
        return self._engine

    def _get_ocr_engine(self):
        if self._ocr_engine is None:
            from .ocr import _load_engine

            log.info("Loading OCR engine (this happens once) ...")
            self._ocr_engine = _load_engine()
        return self._ocr_engine

    def run(self) -> None:
        while True:
            job_id = self.q.get()
            job = self.store.get(job_id)
            if job is None:
                continue
            job.status = "running"
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
                # Clean up the upload temp dir WE created (URLs/feeds have none).
                if job._upload_tmp_dir:
                    shutil.rmtree(job._upload_tmp_dir, ignore_errors=True)

    # -- legacy ASR (byte-stable; guarded by the byte-golden test) -----------

    def _run_asr(self, job: Job) -> None:
        from . import transcribe

        engine = self._get_engine()
        job.transcript = transcribe(
            job._local_path,
            diarize=job.diarize,
            language=job.language,
            min_speakers=job.min_speakers,
            max_speakers=job.max_speakers,
            engine=engine,
        )
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

        work = Path(tempfile.mkdtemp(prefix="transcript-extract-"))
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
                )
            elif job.kind == "video":
                result, asset_files = self._run_video(job, work, transcribe)
            else:
                raise ValueError(f"unknown extraction kind: {job.kind!r}")

            # Identity → ExtractionResult.meta (NEVER Transcript.meta — leak #2).
            result.meta.update({"job_id": job.id, "server_version": _ver})
            # Publish atomically: build staging bundle, rename into place, THEN done.
            self.extraction_store.record(job.id, job.kind, serialize(result), asset_files)
            job.status = "done"
            log.info("Job %s: extraction done (%d assets)", job.id, len(asset_files))
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _run_video(self, job: Job, work: Path, transcribe):
        """Acquire ASR audio from the SAME bestaudio stream the legacy path uses,
        plus a separate capped video stream for frames (plan §B)."""
        from .extract import extract_video
        from .ingest import download_frame_video, is_url

        # ASR: transcribe() downloads bestaudio/best (URL) or decodes the local
        # media — identical to an audio job for this source.
        transcript = transcribe(job._local_path, diarize=job.diarize,
                                language=job.language, engine=self._get_engine())
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
    """Background TTL sweeper for the durable extraction store. Distinct from the
    single Worker thread (which has nothing to evict). Runs on a fixed cadence;
    never evicts a running job or a leased (mid-stream) bundle."""

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

    def run(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            try:
                running = {j.id for j in self.job_store.all()
                           if j.status in ("queued", "running")}
                evicted = self.extraction_store.evict_expired(running_ids=running)
                if evicted:
                    log.info("Janitor evicted %d expired bundle(s).", len(evicted))
                # Also reap staging dirs orphaned by a failed publish mid-session.
                staged = self.extraction_store.gc_staging(running_ids=running)
                if staged:
                    log.info("Janitor reaped %d orphaned staging dir(s).", staged)
            except Exception:  # noqa: BLE001 — never let the sweeper die
                log.exception("Janitor sweep failed")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


def create_app(model: str = "large-v3", device: Optional[str] = None):
    from .formats import FORMATS, render

    token = os.environ.get("TRANSCRIPT_TOKEN")
    if not token:
        log.warning(
            "TRANSCRIPT_TOKEN is not set — the server is running WITHOUT authentication. "
            "Only do this on a trusted, firewalled network."
        )

    from .extraction import KINDS as EXTRACTION_KINDS
    from .extraction_store import ExtractionStore

    store = JobStore()
    extraction_store = ExtractionStore()  # scans existing bundles on construction
    worker = Worker(store, model=model, device=device, extraction_store=extraction_store)
    janitor = Janitor(store, extraction_store)

    app = FastAPI(title="transcript", version="0.1.0")

    @app.on_event("startup")
    def _startup():
        worker.start()
        janitor.start()

    @app.on_event("shutdown")
    def _shutdown():
        janitor.stop()

    def auth(authorization: str | None = Header(default=None)) -> None:
        if not token:
            return
        expected = f"Bearer {token}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Missing or invalid bearer token.")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model, "queued_or_running": [
            j.id for j in store.all() if j.status in ("queued", "running")
        ]}

    @app.post("/jobs")
    async def create_job(
        _: None = Depends(auth),
        url: Optional[str] = Form(default=None),
        file: Optional[UploadFile] = File(default=None),
        diarize: bool = Form(default=True),
        language: Optional[str] = Form(default=None),
        min_speakers: Optional[int] = Form(default=None),
        max_speakers: Optional[int] = Form(default=None),
    ):
        if not url and file is None:
            raise HTTPException(status_code=400, detail="Provide either 'url' or a 'file' upload.")

        job_id = uuid.uuid4().hex[:12]

        upload_tmp_dir = None
        if file is not None:
            upload_tmp_dir = tempfile.mkdtemp(prefix="transcript-upload-")
            dest = Path(upload_tmp_dir) / _safe_upload_name(file.filename)
            with dest.open("wb") as fh:
                shutil.copyfileobj(file.file, fh)
            local_path, source = str(dest), f"upload:{file.filename}"
        else:
            local_path, source = url, url

        job = Job(
            id=job_id,
            source=source,
            diarize=diarize,
            language=language,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        job._local_path = local_path
        job._upload_tmp_dir = upload_tmp_dir
        store.add(job)
        worker.submit(job_id)
        return job.public()

    # The legacy /jobs API is Transcript-ONLY: extraction jobs share the in-memory
    # JobStore but must never be visible or fetchable here (else /jobs/{id}/result
    # would call render() on a None transcript → 500). They live behind /extractions.
    def _asr_job(job_id: str) -> Optional[Job]:
        job = store.get(job_id)
        return job if (job is not None and job.kind is None) else None

    @app.get("/jobs")
    def list_jobs(_: None = Depends(auth)):
        return [j.public() for j in store.all() if j.kind is None]

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str, _: None = Depends(auth)):
        job = _asr_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job.")
        return job.public()

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

    # -- extraction routes (separate envelope; auth required; never the /jobs API) --

    @app.post("/extractions")
    async def create_extraction(
        _: None = Depends(auth),
        kind: str = Form(...),
        url: Optional[str] = Form(default=None),
        file: Optional[UploadFile] = File(default=None),
        diarize: bool = Form(default=True),
        language: Optional[str] = Form(default=None),
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
        # Per-kind source contract — reject mismatches at SUBMIT, not async:
        #  * image_note: upload-only (worker hands the local archive to
        #    archive.extract_images, which needs a path).
        #  * video: a URL or an uploaded media file (podcast selectors are N/A).
        #  * audio_extraction: podcast-only — needs feed_url or an explicit
        #    enclosure_url (a bare URL/file can't prove provenance, plan §C).
        if kind == "image_note" and file is None:
            raise HTTPException(
                status_code=400,
                detail="kind=image_note requires a 'file' upload (a zip/tar archive).",
            )
        if kind == "video" and not url and file is None:
            raise HTTPException(
                status_code=400, detail="kind=video requires 'url' or a 'file' upload.",
            )
        if kind == "audio_extraction" and not feed_url and not enclosure_url:
            raise HTTPException(
                status_code=400,
                detail="kind=audio_extraction requires 'feed_url' (+selector) or "
                "'enclosure_url'.",
            )

        job_id = uuid.uuid4().hex[:12]
        upload_tmp_dir = None
        if file is not None:
            upload_tmp_dir = tempfile.mkdtemp(prefix="transcript-upload-")
            dest = Path(upload_tmp_dir) / _safe_upload_name(file.filename)
            with dest.open("wb") as fh:
                shutil.copyfileobj(file.file, fh)
            local_path, source = str(dest), f"upload:{file.filename}"
        else:
            local_path = source = url or feed_url or enclosure_url

        job = Job(
            id=job_id, source=source, kind=kind, diarize=diarize, language=language,
            frames=frames, cadence_s=cadence_s, feed_url=feed_url,
            episode_guid=episode_guid, episode_url=episode_url,
            episode_title=episode_title, episode_published=episode_published,
            enclosure_url=enclosure_url,
        )
        job._local_path = local_path
        job._upload_tmp_dir = upload_tmp_dir
        store.add(job)
        worker.submit(job_id)
        return job.public_extraction()

    @app.get("/extractions/{job_id}")
    def get_extraction(job_id: str, _: None = Depends(auth)):
        rec = extraction_store.get(job_id)
        # rec present but the on-disk bundle vanished (deleted out-of-band, or a
        # stale index entry) → 410, consistent with /result and /bundle.
        if rec is not None and extraction_store.result_path(job_id).is_file():
            return {"id": job_id, "kind": rec.get("kind"), "status": "done"}
        if rec is not None:
            raise HTTPException(status_code=410, detail="Bundle was evicted or lost.")
        job = store.get(job_id)
        if job is not None and job.kind is not None:
            # In-memory says done but the durable bundle is gone → evicted/lost.
            if job.status == "done":
                raise HTTPException(status_code=410, detail="Bundle was evicted or lost.")
            return job.public_extraction()
        if extraction_store.was_evicted(job_id):  # known-but-evicted (no in-memory job)
            raise HTTPException(status_code=410, detail="Bundle was evicted or lost.")
        raise HTTPException(status_code=404, detail="No such extraction.")

    @app.get("/extractions/{job_id}/result", response_class=PlainTextResponse)
    def get_extraction_result(job_id: str, _: None = Depends(auth)):
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

    @app.get("/extractions/{job_id}/bundle")
    def get_extraction_bundle(job_id: str, _: None = Depends(auth)):
        import os as _os
        import tempfile
        import zipfile

        from fastapi.responses import StreamingResponse

        # Build the zip to a TEMP FILE under the lease (so the asset files can't be
        # evicted mid-build), then stream that file back. We never hold the whole
        # bundle — which can approach the 2 GiB archive cap — in RAM. Zip
        # determinism is NOT required: the consumer hashes `text` + per-asset
        # sha256, not the zip.
        with extraction_store.lease(job_id) as job_dir:
            if job_dir is None:
                # known-but-gone (in-memory done OR the durable index still has it)
                # → 410, consistent with /result and the status route.
                job = store.get(job_id)
                known = ((job is not None and job.kind is not None and job.status == "done")
                         or extraction_store.get(job_id) is not None
                         or extraction_store.was_evicted(job_id))
                if known:
                    raise HTTPException(status_code=410, detail="Bundle was evicted or lost.")
                raise HTTPException(status_code=404, detail="No such extraction bundle.")
            tmp = tempfile.NamedTemporaryFile(prefix="bundle-", suffix=".zip", delete=False)
            tmp.close()
            manifest_path = job_dir / ExtractionStore.MANIFEST_NAME
            try:
                with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
                    for path in sorted(job_dir.rglob("*")):
                        # Exclude the mutable side manifest by EXACT path (not name)
                        # so a stray nested manifest.json is never silently dropped.
                        if path.is_file() and path != manifest_path:
                            zf.write(path, arcname=path.relative_to(job_dir).as_posix())
            except BaseException:
                _os.unlink(tmp.name)  # don't leak the temp file if the build fails
                raise

        def _stream():
            try:
                with open(tmp.name, "rb") as fh:
                    while True:
                        chunk = fh.read(1024 * 1024)
                        if not chunk:
                            break
                        yield chunk
            finally:
                _os.unlink(tmp.name)

        return StreamingResponse(_stream(), media_type="application/zip")

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="transcript-server", description="Transcription HTTP API.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    parser.add_argument("--model", default="large-v3", help="Whisper model (default: large-v3).")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Force device (default: auto).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import uvicorn

    app = create_app(model=args.model, device=args.device)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
