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

log = logging.getLogger("transcript.server")

# ---------------------------------------------------------------------------
# Job model + in-memory store
# ---------------------------------------------------------------------------


@dataclass
class Job:
    id: str
    source: str  # "upload:<name>" or the URL
    status: str = "queued"  # queued | running | done | error
    error: Optional[str] = None
    # Pipeline options
    diarize: bool = True
    language: Optional[str] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    # Filled when done
    transcript: object = field(default=None, repr=False)
    # Internal: the local path to process (uploaded temp file or the URL)
    _local_path: Optional[str] = field(default=None, repr=False)

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
    def __init__(self, store: JobStore, model: str, device: Optional[str]):
        super().__init__(daemon=True)
        self.store = store
        self.model = model
        self.device = device
        self.q: "queue.Queue[str]" = queue.Queue()
        self._engine = None

    def submit(self, job_id: str) -> None:
        self.q.put(job_id)

    def _get_engine(self):
        if self._engine is None:
            from .engine import TranscriptionEngine

            log.info("Loading model '%s' (this happens once) ...", self.model)
            self._engine = TranscriptionEngine(model=self.model, device=self.device)
            log.info("Model ready on device=%s.", self._engine.device)
        return self._engine

    def run(self) -> None:
        from . import transcribe

        while True:
            job_id = self.q.get()
            job = self.store.get(job_id)
            if job is None:
                continue
            job.status = "running"
            log.info("Job %s: running (%s)", job.id, job.source)
            try:
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
            except Exception as exc:  # noqa: BLE001 — surface to client
                job.status = "error"
                job.error = str(exc)
                log.exception("Job %s failed", job.id)
            finally:
                # Clean up an uploaded temp file (URLs have no temp upload).
                if job._local_path and job._local_path.startswith(tempfile.gettempdir()):
                    shutil.rmtree(Path(job._local_path).parent, ignore_errors=True)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


def create_app(model: str = "large-v3", device: Optional[str] = None):
    from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
    from fastapi.responses import PlainTextResponse

    from .formats import FORMATS, render

    token = os.environ.get("TRANSCRIPT_TOKEN")
    if not token:
        log.warning(
            "TRANSCRIPT_TOKEN is not set — the server is running WITHOUT authentication. "
            "Only do this on a trusted, firewalled network."
        )

    store = JobStore()
    worker = Worker(store, model=model, device=device)

    app = FastAPI(title="transcript", version="0.1.0")

    @app.on_event("startup")
    def _startup():
        worker.start()

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

        if file is not None:
            tmp_dir = Path(tempfile.mkdtemp(prefix="transcript-upload-"))
            dest = tmp_dir / (file.filename or "upload.bin")
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
        store.add(job)
        worker.submit(job_id)
        return job.public()

    @app.get("/jobs")
    def list_jobs(_: None = Depends(auth)):
        return [j.public() for j in store.all()]

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str, _: None = Depends(auth)):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job.")
        return job.public()

    @app.get("/jobs/{job_id}/result", response_class=PlainTextResponse)
    def get_result(job_id: str, format: str = "txt", _: None = Depends(auth)):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job.")
        if job.status == "error":
            raise HTTPException(status_code=409, detail=f"Job failed: {job.error}")
        if job.status != "done":
            raise HTTPException(status_code=409, detail=f"Job not finished (status: {job.status}).")
        if format not in FORMATS:
            raise HTTPException(status_code=400, detail=f"format must be one of {FORMATS}")
        return render(job.transcript, format)

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
