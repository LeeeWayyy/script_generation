import asyncio
import io
import os
import shutil
import socket
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from transcript.extraction_store import ExtractionStore
from transcript.ingest import SsrfError, assert_public_url, is_url
from transcript.types import Transcript


def _app(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSCRIPT_DATA_DIR", str(tmp_path / "store"))
    import transcript.server as server

    return server.create_app()


def test_public_url_guard_rejects_cgnat_and_parses_uppercase(monkeypatch):
    monkeypatch.delenv("TRANSCRIPT_ALLOW_PRIVATE_FETCH", raising=False)
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("100.64.0.1", 80))],
    )
    assert is_url("HTTP://example.com/media")
    with pytest.raises(SsrfError, match="non-public"):
        assert_public_url("HTTP://example.com/media")


def test_staged_upload_provenance_uses_sanitized_name():
    import transcript.server as server

    upload = SimpleNamespace(filename="../bad\nname.mp3", file=io.BytesIO(b"audio"))
    path, source, directory = server._stage_upload(upload)
    try:
        assert source == "upload:upload.bin"
        assert os.path.basename(path) == "upload.bin"
    finally:
        shutil.rmtree(directory)


def test_submit_validation_rejects_paths_conflicts_and_bad_speakers(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSCRIPT_ALLOW_PRIVATE_FETCH", "1")
    with TestClient(_app(monkeypatch, tmp_path)) as client:
        assert client.post("/jobs", data={"url": "/etc/passwd"}).status_code == 400
        both = client.post(
            "/jobs", data={"url": "https://example.com/x"},
            files={"file": ("x.mp3", b"x")},
        )
        assert both.status_code == 400 and "exactly one" in both.text
        assert client.post(
            "/jobs", data={"min_speakers": "0"}, files={"file": ("x.mp3", b"x")},
        ).status_code == 400
        assert client.post(
            "/jobs", data={"min_speakers": "3", "max_speakers": "2"},
            files={"file": ("x.mp3", b"x")},
        ).status_code == 400
        speaker_without_diarize = client.post(
            "/jobs", data={"diarize": "false", "min_speakers": "1"},
            files={"file": ("x.mp3", b"x")},
        )
        assert speaker_without_diarize.status_code == 400

        conflict = client.post("/extractions", data={
            "kind": "audio_extraction", "feed_url": "https://example.com/feed",
            "enclosure_url": "https://example.com/ep.mp3", "episode_guid": "g",
        })
        assert conflict.status_code == 400 and "exactly one" in conflict.text
        missing = client.post("/extractions", data={
            "kind": "audio_extraction", "feed_url": "https://example.com/feed",
        })
        assert missing.status_code == 400 and "requires" in missing.text
        partial = client.post("/extractions", data={
            "kind": "audio_extraction", "feed_url": "https://example.com/feed",
            "episode_title": "Episode",
        })
        assert partial.status_code == 400 and "together" in partial.text
        ignored_selector = client.post("/extractions", data={
            "kind": "audio_extraction", "feed_url": "https://example.com/feed",
            "episode_guid": "g", "episode_title": "Episode",
            "episode_published": "2026-01-01",
        })
        assert ignored_selector.status_code == 400 and "cannot accompany" in ignored_selector.text
        invalid = client.post("/extractions", data={
            "kind": "audio_extraction", "enclosure_url": "file:///etc/passwd",
        })
        assert invalid.status_code == 400 and "scheme" in invalid.text
        for option in (
            {"diarize": "false"}, {"detect_music": "true"}, {"min_speakers": "1"},
            {"frames": "true"}, {"cadence_s": "5"}, {"language": "en"},
        ):
            rejected = client.post(
                "/extractions", data={"kind": "image_note", **option},
                files={"file": ("cards.zip", b"x")},
            )
            assert rejected.status_code == 400 and "not applicable" in rejected.text
        assert client.post(
            "/extractions", data={"kind": "video", "cadence_s": "5"},
            files={"file": ("x.mp4", b"x")},
        ).status_code == 400
        for cadence in ("nan", "inf", "-inf"):
            assert client.post(
                "/extractions",
                data={"kind": "video", "frames": "true", "cadence_s": cadence},
                files={"file": ("x.mp4", b"x")},
            ).status_code == 400
        assert client.post("/extractions", data={
            "kind": "audio_extraction", "enclosure_url": "https://example.com/e.mp3",
            "frames": "true",
        }).status_code == 400


@pytest.mark.parametrize("name,value", [
    ("TRANSCRIPT_MAX_QUEUE_SIZE", "0"),
    ("TRANSCRIPT_MAX_TERMINAL_JOBS", "0"),
    ("TRANSCRIPT_MAX_CONCURRENT_BUNDLES", "0"),
    ("TRANSCRIPT_JOB_TTL_SECONDS", "nan"),
    ("TRANSCRIPT_JANITOR_INTERVAL_SECONDS", "inf"),
])
def test_server_limits_must_be_finite_and_positive(monkeypatch, tmp_path, name, value):
    monkeypatch.setenv("TRANSCRIPT_DATA_DIR", str(tmp_path / "store"))
    monkeypatch.setenv(name, value)
    import transcript.server as server

    with pytest.raises(ValueError, match="must be positive"):
        server.create_app()


def test_upload_limit_must_be_positive(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSCRIPT_DATA_DIR", str(tmp_path / "store"))
    import transcript.server as server

    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 0)
    with pytest.raises(ValueError, match="must be positive"):
        server.create_app()


def test_request_body_limit_counts_streamed_chunks_without_content_length():
    import transcript.server as server

    called = False

    async def inner(scope, receive, send):
        nonlocal called
        while True:
            message = await receive()
            if not message.get("more_body"):
                break
        called = True

    messages = iter([
        {"type": "http.request", "body": b"1234", "more_body": True},
        {"type": "http.request", "body": b"5678", "more_body": False},
    ])
    sent = []

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    middleware = server.RequestBodyLimitMiddleware(inner, max_bytes=6)
    asyncio.run(middleware(
        {"type": "http", "method": "POST", "path": "/", "headers": []}, receive, send,
    ))
    assert not called
    assert sent[0]["status"] == 413


def test_bounded_queue_reports_position_and_cleans_rejected_upload(
    monkeypatch, tmp_path,
):
    import transcript.server as server

    monkeypatch.setenv("TRANSCRIPT_DATA_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("TRANSCRIPT_MAX_QUEUE_SIZE", "1")
    gate = threading.Event()
    original_run = server.Worker.run

    def gated_run(worker):
        gate.wait()
        original_run(worker)

    monkeypatch.setattr(server.Worker, "run", gated_run)
    staged = []

    def fake_stage(file):
        directory = tmp_path / f"upload-{len(staged) + 1}"
        directory.mkdir()
        path = directory / file.filename
        path.write_bytes(b"x")
        staged.append(directory)
        return str(path), f"upload:{file.filename}", str(directory)

    monkeypatch.setattr(server, "_stage_upload", fake_stage)
    app = server.create_app()
    with TestClient(app) as client:
        try:
            first = client.post("/extractions", data={
                "kind": "video", "detect_music": "true",
                "min_speakers": "2", "max_speakers": "3",
            }, files={"file": ("first.mp4", b"x")})
            assert first.status_code == 200, first.text
            job_id = first.json()["id"]
            status = client.get(f"/extractions/{job_id}").json()
            assert status["queue_position"] == 1
            assert status["detect_music"] is True
            job = app.state.job_store.get(job_id)
            assert (job.min_speakers, job.max_speakers) == (2, 3)

            full = client.post(
                "/jobs", files={"file": ("second.mp3", b"x")},
            )
            assert full.status_code == 503
            assert not staged[1].exists()
            assert client.delete(f"/extractions/{job_id}").status_code == 204
            assert not staged[0].exists()
        finally:
            gate.set()


def test_job_store_prunes_terminal_ttl_and_count_without_running_jobs():
    import transcript.server as server

    store = server.JobStore(max_terminal_jobs=1, terminal_ttl_s=10)
    old = server.Job("old", "x", status="done", finished_at=1)
    newer = server.Job("newer", "x", status="done", finished_at=95)
    newest = server.Job("newest", "x", status="error", finished_at=96)
    running = server.Job("running", "x", status="running", created_at=1)
    unstamped = server.Job("unstamped", "x", status="done", created_at=1)
    for job in (old, newer, newest, running, unstamped):
        store.add(job)
    assert store.prune(now=100) == ["newer", "old"]
    assert {job.id for job in store.all()} == {"newest", "running", "unstamped"}


def test_worker_stops_cleanly_and_janitor_sweeps_temp(monkeypatch, tmp_path):
    import transcript.server as server

    store = server.JobStore()
    worker = server.Worker(store, "tiny", "cpu", max_queue_size=1)
    worker.start()
    worker.stop()
    worker.join(timeout=2)
    assert not worker.is_alive()

    upload = tmp_path / "transcript-upload-live"
    work = tmp_path / "transcript-extract-live"
    enclosure = tmp_path / "enclosure-live"
    orphan = tmp_path / "transcript-upload-orphan"
    bundle = tmp_path / "bundle-live.zip"
    for path in (upload, work, enclosure, orphan):
        path.mkdir()
        os.utime(path, (1, 1))
    bundle.write_bytes(b"zip")
    os.utime(bundle, (1, 1))
    queued = server.Job("queued", "upload:a", _upload_tmp_dir=str(upload))
    running = server.Job("running", "upload:b", status="running", kind="video",
                         _work_tmp_dir=str(work))
    podcast = server.Job("podcast", "https://example.com/feed", status="running",
                         kind="audio_extraction")
    for job in (queued, running, podcast):
        store.add(job)
    monkeypatch.setattr(server.tempfile, "gettempdir", lambda: str(tmp_path))
    with server._ACTIVE_BUNDLE_TEMPS_LOCK:
        server._ACTIVE_BUNDLE_TEMPS.add(str(bundle))
    janitor = server.Janitor(store, ExtractionStore(tmp_path / "extractions"))
    try:
        janitor.sweep()
        assert all(path.exists() for path in (upload, work, enclosure, bundle))
        assert not orphan.exists()
    finally:
        with server._ACTIVE_BUNDLE_TEMPS_LOCK:
            server._ACTIVE_BUNDLE_TEMPS.discard(str(bundle))


def test_ocr_unavailable_sentinel_is_cached(monkeypatch):
    import transcript.server as server
    from transcript.extract import OCR_UNAVAILABLE
    from transcript.ocr import OcrUnavailableError

    calls = []

    def unavailable():
        calls.append(True)
        raise OcrUnavailableError("weights missing")

    monkeypatch.setattr("transcript.ocr._load_engine", unavailable)
    worker = server.Worker(server.JobStore(), "tiny", "cpu")
    assert worker._get_ocr_engine() is OCR_UNAVAILABLE
    assert worker._get_ocr_engine() is OCR_UNAVAILABLE
    assert calls == [True]


def test_delete_legacy_result_and_durable_extraction(monkeypatch, tmp_path):
    import transcript.server as server

    app = _app(monkeypatch, tmp_path)
    legacy_id = "111111111111"
    extraction_id = "222222222222"
    legacy = server.Job(legacy_id, "upload:a.mp3", status="done", transcript=Transcript())
    extraction = server.Job(extraction_id, "upload:a.zip", status="done", kind="image_note")
    app.state.job_store.add(legacy)
    app.state.job_store.add(extraction)
    asset = tmp_path / "asset.jpg"
    asset.write_bytes(b"jpg")
    app.state.extraction_store.record(
        extraction_id, "image_note", "{}", [("assets/asset.jpg", asset)],
    )

    with TestClient(app) as client:
        assert client.delete(f"/jobs/{legacy_id}").status_code == 204
        assert client.get(f"/jobs/{legacy_id}").status_code == 404
        with app.state.extraction_store.lease(extraction_id):
            assert client.delete(f"/extractions/{extraction_id}").status_code == 409
        assert client.delete(f"/extractions/{extraction_id}").status_code == 204
        assert client.get(f"/extractions/{extraction_id}").status_code == 410


def test_music_and_speaker_options_reach_both_extraction_transcribe_paths(
    monkeypatch, tmp_path,
):
    import transcript.server as server
    from transcript.extraction import ExtractionResult

    captured = []

    def fake_audio(**kwargs):
        captured.append(kwargs)
        return ExtractionResult(kind="audio_extraction", text=""), []

    monkeypatch.setattr("transcript.extract.extract_audio_extraction", fake_audio)
    published = []

    class Store:
        def record(self, *args, **kwargs):
            published.append((args, kwargs))

    worker = server.Worker(server.JobStore(), "tiny", "cpu", extraction_store=Store())
    monkeypatch.setattr(worker, "_get_engine", lambda: "engine")
    job = server.Job(
        "333333333333", "https://example.com/feed", kind="audio_extraction",
        feed_url="https://example.com/feed", episode_guid="g", detect_music=True,
        min_speakers=2, max_speakers=4,
    )
    worker._run_extraction(job)
    assert captured[0]["detect_music"] is True
    assert (captured[0]["min_speakers"], captured[0]["max_speakers"]) == (2, 4)
    assert published

    video_calls = []

    def fake_transcribe(source, **kwargs):
        assert os.path.isdir(kwargs["work_dir"])
        video_calls.append(kwargs)
        return Transcript()

    monkeypatch.setattr(
        "transcript.extract.extract_video",
        lambda **kwargs: (ExtractionResult(kind="video", text=""), []),
    )
    video = server.Job(
        "444444444444", "upload:clip.mp4", kind="video", detect_music=True,
        min_speakers=3, max_speakers=5,
    )
    video._local_path = str(tmp_path / "clip.mp4")
    worker._run_video(video, tmp_path, fake_transcribe)
    assert video_calls[0]["detect_music"] is True
    assert (video_calls[0]["min_speakers"], video_calls[0]["max_speakers"]) == (3, 5)
