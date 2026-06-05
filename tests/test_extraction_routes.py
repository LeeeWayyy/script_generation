"""End-to-end extraction route: upload archive → poll → fetch+verify bundle.

OCR is monkeypatched (no PaddleOCR). Exercises the separate /extractions route,
the durable store, auth, the 404/410 contract, and that the bundle's result.json
is byte-identical to the /result route.
"""

import io
import time
import zipfile


from transcript.extract_remote import unpack_and_verify
from transcript.ocr import OcrResult


def _fake_ocr(image_path, engine=None):
    # Deterministic OCR keyed off the basename so card text is predictable.
    return OcrResult(ocr_text=f"text-of-{image_path.name}", confidence=0.9,
                     width=100, height=50, blocks=[])


def _make_app(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSCRIPT_DATA_DIR", str(tmp_path / "store"))
    monkeypatch.setattr("transcript.ocr._load_engine", lambda: "FAKE_OCR")
    monkeypatch.setattr("transcript.ocr.run_ocr", _fake_ocr)
    from fastapi.testclient import TestClient
    import transcript.server as server
    return TestClient(server.create_app())


def _zip_bytes(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _run_image_note(client, zip_bytes):
    r = client.post("/extractions", data={"kind": "image_note"},
                    files={"file": ("export.zip", zip_bytes)})
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    for _ in range(200):
        s = client.get(f"/extractions/{job_id}").json()
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.02)
    assert s["status"] == "done", s
    return job_id


def test_image_note_end_to_end(monkeypatch, tmp_path):
    with _make_app(monkeypatch, tmp_path) as client:
        zb = _zip_bytes({"b/2.jpg": b"two", "a/1.jpg": b"one"})
        job_id = _run_image_note(client, zb)

        # /result is the durable envelope JSON.
        rr = client.get(f"/extractions/{job_id}/result")
        assert rr.status_code == 200
        envelope = rr.json()
        assert envelope["kind"] == "image_note"
        # Byte-sorted ordering drives 1-based card numbering in the text.
        assert envelope["text"] == "## card 1\ntext-of-1.jpg\n\n## card 2\ntext-of-2.jpg\n"
        assert [c["source_filename"] for c in envelope["cards"]] == ["1.jpg", "2.jpg"]

        # /bundle streams a verifiable zip whose result.json == the /result bytes.
        rb = client.get(f"/extractions/{job_id}/bundle")
        assert rb.status_code == 200
        out = unpack_and_verify(rb.content, tmp_path / "unpacked")
        assert out == envelope
        bundle_result = (tmp_path / "unpacked" / "result.json").read_text()
        assert bundle_result == rr.text  # byte-identical


def test_unknown_extraction_is_404(monkeypatch, tmp_path):
    with _make_app(monkeypatch, tmp_path) as client:
        assert client.get("/extractions/nope").status_code == 404
        assert client.get("/extractions/nope/result").status_code == 404
        assert client.get("/extractions/nope/bundle").status_code == 404


def test_image_note_url_rejected_at_submit(monkeypatch, tmp_path):
    # image_note is upload-only (the worker needs a local archive path); a URL
    # must be rejected at submit, not fail the job asynchronously.
    with _make_app(monkeypatch, tmp_path) as client:
        r = client.post("/extractions",
                        data={"kind": "image_note", "url": "https://x/export.zip"})
        assert r.status_code == 400 and "file" in r.json()["detail"].lower()


def test_evicted_bundle_returns_410(monkeypatch, tmp_path):
    with _make_app(monkeypatch, tmp_path) as client:
        job_id = _run_image_note(client, _zip_bytes({"a.jpg": b"x"}))
        # Simulate loss of the durable bundle (TTL eviction / lost on restart)
        # while the in-memory job still reports done → 410 Gone (not 404).
        import shutil
        shutil.rmtree(tmp_path / "store" / job_id)
        assert client.get(f"/extractions/{job_id}").status_code == 410  # status endpoint too
        assert client.get(f"/extractions/{job_id}/result").status_code == 410
        assert client.get(f"/extractions/{job_id}/bundle").status_code == 410


def _poll_done(client, job_id):
    for _ in range(200):
        s = client.get(f"/extractions/{job_id}").json()
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.02)
    assert s["status"] == "done", s


def test_audio_extraction_podcast_provenance(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSCRIPT_DATA_DIR", str(tmp_path / "store"))
    from transcript.types import Segment, Transcript
    from transcript.podcast import PodcastResolution
    import transcript.server as server

    def fake_transcribe(source, *, engine=None, **kw):
        # ASR-side meta is lifted onto the envelope; podcast fields are added on top.
        t = Transcript(segments=[Segment(text="Café episode.", speaker="SPEAKER_00")],
                       language="en")
        t.meta.update({"model": "large-v3", "source": source})
        return t

    def fake_resolve(feed_url, *, episode_guid=None, episode_url=None, **kw):
        return PodcastResolution(
            enclosure_url="https://cdn.example.com/ep1.mp3", resolution_source="feed_parse",
            feed_url=feed_url, episode_guid=episode_guid, published="2024-01-01")

    from pathlib import Path
    from transcript.podcast import EnclosureDownload
    monkeypatch.setattr(server.Worker, "_get_engine", lambda self: "ENG")
    monkeypatch.setattr("transcript.transcribe", fake_transcribe)
    monkeypatch.setattr("transcript.podcast.resolve_podcast", fake_resolve)
    monkeypatch.setattr("transcript.podcast.download_enclosure",
                        lambda url, dest, **k: EnclosureDownload(
                            path=Path("/tmp/fake-enc.bin"), final_url=url,
                            content_length=123, downloaded_size=123, ok=True))

    from fastapi.testclient import TestClient
    with TestClient(server.create_app()) as client:
        r = client.post("/extractions", data={
            "kind": "audio_extraction", "feed_url": "https://feed.example.com/rss",
            "episode_guid": "guid-1"})
        assert r.status_code == 200, r.text
        job_id = r.json()["id"]
        _poll_done(client, job_id)
        env = client.get(f"/extractions/{job_id}/result").json()

    assert env["kind"] == "audio_extraction"
    assert env["text"] == "SPEAKER_00: Café episode.\n"  # NFC, speaker prefix
    assert env["meta"]["feed_url"] == "https://feed.example.com/rss"
    assert env["meta"]["episode_guid"] == "guid-1"
    assert env["meta"]["enclosure_url"] == "https://cdn.example.com/ep1.mp3"
    assert env["meta"]["resolution_source"] == "feed_parse"
    assert env["meta"]["content_length"] == 123  # download audit recorded
    assert env["meta"]["downloaded_size"] == 123
    assert "source" not in env["meta"]  # transient temp path must not leak
    assert len(env["segments"]) == 1


def test_video_frames_and_audio_stream_pinning(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSCRIPT_DATA_DIR", str(tmp_path / "store"))
    from transcript.types import Segment, Transcript
    from transcript.frames import FrameAsset
    import transcript.server as server

    def fake_transcribe(source, *, engine=None, **kw):
        t = Transcript(segments=[Segment(text="hello", speaker=None)], language="en")
        # `source` here is the server temp path — it must NOT reach the envelope.
        t.meta.update({"model": "large-v3", "selected_format": "140", "source": source})
        return t

    def fake_extract_frames(video_path, dest_dir, *, cadence_s, **kw):
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for i in range(2):
            p = dest_dir / f"frame-{i:06d}.jpg"
            p.write_bytes(f"frame{i}".encode())
            out.append(FrameAsset(frame_id=i, timecode=float(i * cadence_s), path=p))
        return out

    monkeypatch.setattr(server.Worker, "_get_engine", lambda self: "ENG")
    monkeypatch.setattr(server.Worker, "_get_ocr_engine", lambda self: "OCR")
    monkeypatch.setattr("transcript.transcribe", fake_transcribe)
    monkeypatch.setattr("transcript.frames.extract_frames", fake_extract_frames)
    monkeypatch.setattr("transcript.ocr.run_ocr", _fake_ocr)

    from fastapi.testclient import TestClient
    with TestClient(server.create_app()) as client:
        # --frames is the opt-in switch for video frame extraction.
        r = client.post("/extractions", data={"kind": "video", "frames": "true"},
                        files={"file": ("clip.mp4", b"fakevideo")})
        assert r.status_code == 200, r.text
        job_id = r.json()["id"]
        _poll_done(client, job_id)
        rr = client.get(f"/extractions/{job_id}/result")
        env = rr.json()
        rb = client.get(f"/extractions/{job_id}/bundle")
        out = unpack_and_verify(rb.content, tmp_path / "vunpack")

    assert env["kind"] == "video"
    assert env["text"] == "hello\n"  # frame OCR is NOT merged into audio text
    assert len(env["frames"]) == 2
    assert env["frames"][0]["ocr_text"].startswith("text-of-")  # OCR kept in frames[]
    # ASR audio came from the bestaudio stream (selected_format), recorded as
    # selected_audio_format on the envelope.
    assert env["meta"]["selected_audio_format"] == "140"
    # The legacy singular selected_format must NOT leak into the video envelope,
    # nor the transient server-side `source` temp path.
    assert "selected_format" not in env["meta"]
    assert "source" not in env["meta"]
    assert env["meta"]["frame_count"] == 2
    assert out == env  # bundle result.json == /result bytes


def test_video_without_frames_flag_skips_frame_extraction(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSCRIPT_DATA_DIR", str(tmp_path / "store"))
    from transcript.types import Segment, Transcript
    import transcript.server as server

    def fake_transcribe(source, *, engine=None, **kw):
        return Transcript(segments=[Segment(text="hello")], language="en")

    def boom(*a, **k):
        raise AssertionError("extract_frames must not run without --frames")

    monkeypatch.setattr(server.Worker, "_get_engine", lambda self: "ENG")
    monkeypatch.setattr("transcript.transcribe", fake_transcribe)
    monkeypatch.setattr("transcript.frames.extract_frames", boom)

    from fastapi.testclient import TestClient
    with TestClient(server.create_app()) as client:
        r = client.post("/extractions", data={"kind": "video"},  # no frames flag
                        files={"file": ("clip.mp4", b"fakevideo")})
        job_id = r.json()["id"]
        _poll_done(client, job_id)
        env = client.get(f"/extractions/{job_id}/result").json()
    assert env["kind"] == "video"
    assert env["frames"] == [] and env["meta"]["frame_count"] == 0
    assert "frame_policy" not in env["meta"]
    # The provenance recipe must not name absent keys.
    assert "frame_policy" not in env["meta"]["_provenance"]["recipe"]


def test_auth_required_on_extraction_routes(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSCRIPT_TOKEN", "secret")
    with _make_app(monkeypatch, tmp_path) as client:
        # No Authorization header → 401 on ALL new endpoints.
        assert client.post("/extractions", data={"kind": "image_note"},
                           files={"file": ("e.zip", _zip_bytes({"a.jpg": b"x"}))}
                           ).status_code == 401
        assert client.get("/extractions/x").status_code == 401
        assert client.get("/extractions/x/result").status_code == 401
        assert client.get("/extractions/x/bundle").status_code == 401


def test_extraction_jobs_invisible_on_legacy_jobs_route(monkeypatch, tmp_path):
    # Leak #1: the legacy /jobs API must stay Transcript-only. An extraction job
    # id must 404 on /jobs/{id} and /jobs/{id}/result (never 500 on render(None)).
    with _make_app(monkeypatch, tmp_path) as client:
        job_id = _run_image_note(client, _zip_bytes({"a.jpg": b"x"}))
        assert client.get(f"/jobs/{job_id}").status_code == 404
        assert client.get(f"/jobs/{job_id}/result", params={"format": "json"}).status_code == 404
        assert job_id not in [j["id"] for j in client.get("/jobs").json()]


def test_unsafe_upload_filename_is_sanitized(monkeypatch, tmp_path):
    import transcript.server as server
    # A traversal filename must not escape the temp dir.
    assert server._safe_upload_name("../../etc/passwd") == "passwd"
    assert server._safe_upload_name("/abs/x.zip") == "x.zip"
    assert server._safe_upload_name("..") == "upload.bin"
    assert server._safe_upload_name("") == "upload.bin"
    assert server._safe_upload_name("ok.zip") == "ok.zip"


def test_audio_extraction_explicit_enclosure_is_user_supplied(monkeypatch, tmp_path):
    # An explicit enclosure URL is the only no-feed path that's allowed — recorded
    # honestly as resolution_source=user_supplied (never a minted guid).
    monkeypatch.setenv("TRANSCRIPT_DATA_DIR", str(tmp_path / "store"))
    from transcript.types import Segment, Transcript
    import transcript.server as server

    def fake_transcribe(source, *, engine=None, **kw):
        assert source == "https://cdn.example.com/ep1.mp3"  # the enclosure, downloaded
        return Transcript(segments=[Segment(text="hi")], language="en")

    from transcript.podcast import EnclosureDownload
    monkeypatch.setattr(server.Worker, "_get_engine", lambda self: "ENG")
    monkeypatch.setattr("transcript.transcribe", fake_transcribe)
    # Direct download "fails" → fall back to transcribing the enclosure URL itself.
    monkeypatch.setattr("transcript.podcast.download_enclosure",
                        lambda url, dest, **k: EnclosureDownload(ok=False))

    from fastapi.testclient import TestClient
    with TestClient(server.create_app()) as client:
        r = client.post("/extractions", data={
            "kind": "audio_extraction", "enclosure_url": "https://cdn.example.com/ep1.mp3"})
        assert r.status_code == 200, r.text
        job_id = r.json()["id"]
        _poll_done(client, job_id)
        env = client.get(f"/extractions/{job_id}/result").json()
    assert env["meta"]["resolution_source"] == "user_supplied"
    assert env["meta"]["episode_guid"] is None
    assert env["meta"]["enclosure_url"] == "https://cdn.example.com/ep1.mp3"


def test_audio_extraction_bare_url_rejected_at_submit(monkeypatch, tmp_path):
    # A bare page URL can't prove podcast provenance — rejected at SUBMIT (400),
    # not minted (plan §C "never silently mint weak provenance").
    with _make_app(monkeypatch, tmp_path) as client:
        r = client.post("/extractions", data={"kind": "audio_extraction",
                                              "url": "https://podcast.example.com/page"})
        assert r.status_code == 400
        # A file-only audio_extraction is likewise rejected (no feed identity).
        r2 = client.post("/extractions", data={"kind": "audio_extraction"},
                         files={"file": ("ep.mp3", b"x")})
        assert r2.status_code == 400


def test_video_with_feed_url_only_rejected_at_submit(monkeypatch, tmp_path):
    # kind=video needs url or a file; podcast selectors are not a valid source.
    with _make_app(monkeypatch, tmp_path) as client:
        r = client.post("/extractions", data={"kind": "video",
                                              "feed_url": "https://f/rss"})
        assert r.status_code == 400


def test_audio_extraction_structured_error_reason_surfaced(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSCRIPT_DATA_DIR", str(tmp_path / "store"))
    from transcript.podcast import PodcastResolutionError
    import transcript.server as server

    def fake_resolve(feed_url, *, episode_guid=None, episode_url=None, **kw):
        raise PodcastResolutionError("ambiguous", "GUID duplicated in feed")

    monkeypatch.setattr(server.Worker, "_get_engine", lambda self: "ENG")
    monkeypatch.setattr("transcript.podcast.resolve_podcast", fake_resolve)

    from fastapi.testclient import TestClient
    with TestClient(server.create_app()) as client:
        r = client.post("/extractions", data={"kind": "audio_extraction",
                                              "feed_url": "https://f/rss",
                                              "episode_guid": "g"})
        job_id = r.json()["id"]
        for _ in range(200):
            s = client.get(f"/extractions/{job_id}").json()
            if s["status"] in ("done", "error"):
                break
            time.sleep(0.02)
        assert s["status"] == "error"
        assert s["error_reason"] == "ambiguous"  # distinguishable from stale_selector
        rr = client.get(f"/extractions/{job_id}/result")
        assert rr.status_code == 409 and "ambiguous" in rr.text
