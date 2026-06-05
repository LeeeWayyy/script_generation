"""Shared harness that drives the REAL production path (transcribe() + the
Worker.run meta-stamp) through the public ``/jobs/{id}/result`` route with the
ML/network sites monkeypatched. Used by the byte-golden meta test."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from transcript.types import Segment, Transcript, Word

FIXED_UUID = uuid.UUID(int=0x0123456789ABCDEF0123456789ABCDEF)
FIXED_JOB_ID = FIXED_UUID.hex[:12]


class FakeEngine:
    """Stands in for TranscriptionEngine — mirrors engine.run()'s meta block so
    the four-site insertion order is exercised end to end without whisperx."""

    model_name = "large-v3"
    device = "cpu"
    compute_type = "int8"

    def run(self, audio_path, *, diarize, language, min_speakers, max_speakers, align):
        segs = [
            Segment(text="Hello there.", start=0.0, end=1.5, speaker=None,
                    words=[Word("Hello", 0.0, 0.5, 0.9, None)]),
            Segment(text="General Kenobi.", start=1.6, end=3.2, speaker=None),
        ]
        t = Transcript(segments=segs, language="en")
        ends = [s.end for s in t.segments if s.end is not None]
        # Replicates engine.run's meta.update — the first of the four sites.
        t.meta.update({
            "align_requested": align,
            "align_succeeded": True if align else None,
            "diarize_requested": diarize,
            "diarize_succeeded": (any(s.speaker for s in t.segments) if diarize else None),
            "duration_s": (max(ends) if ends else None),
            "whisperx_version": None,
            "pyannote_version": None,
        })
        return t


def install_patches(monkeypatch, *, info_json: dict | None):
    """Patch the resolve/extract/version/engine/uuid sites for determinism."""
    import transcript
    import transcript.server as server

    def fake_resolve(source, work_path):
        work_path = Path(work_path)
        work_path.mkdir(parents=True, exist_ok=True)
        media = work_path / "GOLDEN.mp4"
        media.write_bytes(b"")
        if info_json is not None:
            (work_path / "GOLDEN.info.json").write_text(json.dumps(info_json), encoding="utf-8")
        return media

    def fake_extract(media, work_path):
        wav = Path(work_path) / "GOLDEN.16k.wav"
        wav.write_bytes(b"")
        return wav

    monkeypatch.setattr(transcript, "resolve_source", fake_resolve)
    monkeypatch.setattr(transcript, "extract_audio", fake_extract)
    monkeypatch.setattr(transcript, "_ffmpeg_version", lambda: "6.0")

    import importlib.metadata as md
    real_version = md.version

    def fake_version(name):
        if name == "yt-dlp":
            return "2024.12.13"
        return real_version(name)

    monkeypatch.setattr(md, "version", fake_version)

    engine = FakeEngine()
    monkeypatch.setattr(server.Worker, "_get_engine", lambda self: engine)
    monkeypatch.setattr(server.uuid, "uuid4", lambda: FIXED_UUID)

    import tempfile
    monkeypatch.setenv("TRANSCRIPT_DATA_DIR", tempfile.mkdtemp(prefix="transcript-test-store-"))


def run_job_get_json(monkeypatch, *, source, info_json):
    """Submit `source`, wait for completion, return the raw /result?format=json bytes."""
    from fastapi.testclient import TestClient

    import transcript.server as server

    install_patches(monkeypatch, info_json=info_json)

    app = server.create_app()
    with TestClient(app) as client:
        r = client.post("/jobs", data={"url": source, "diarize": "false"})
        assert r.status_code == 200, r.text
        job_id = r.json()["id"]
        for _ in range(200):
            s = client.get(f"/jobs/{job_id}").json()
            if s["status"] in ("done", "error"):
                break
            time.sleep(0.02)
        assert s["status"] == "done", s
        rr = client.get(f"/jobs/{job_id}/result", params={"format": "json"})
        assert rr.status_code == 200, rr.text
        return rr.text
