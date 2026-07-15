"""Durable extraction store: atomic publish, TTL eviction, restart scan, crash GC."""


from transcript.extraction_store import ExtractionStore


def _publish(store, job_id="job1", text="{}", now=1000.0):
    asset = store.root / "_src" / "card-000.jpg"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_bytes(b"imgbytes")
    store.record(job_id, "image_note", text, [("assets/card-000.jpg", asset)], now=now)


def test_record_publishes_result_and_assets(tmp_path):
    store = ExtractionStore(root=tmp_path, ttl_s=100)
    _publish(store, text='{"kind":"image_note"}')
    assert store.result_path("job1").read_text() == '{"kind":"image_note"}'
    assert (store.root / "job1" / "assets" / "card-000.jpg").read_bytes() == b"imgbytes"
    # The mutable manifest lives beside (not inside) the immutable result.json.
    assert (store.root / "job1" / "manifest.json").is_file()


def test_read_result_bumps_last_access(tmp_path):
    store = ExtractionStore(root=tmp_path, ttl_s=100)
    _publish(store, now=1000.0)
    assert store.get("job1")["last_access"] == 1000.0
    store.read_result("job1")  # bumps to wall-clock now (>> 1000)
    assert store.get("job1")["last_access"] > 1000.0


def test_eviction_respects_ttl_and_running(tmp_path):
    store = ExtractionStore(root=tmp_path, ttl_s=100)
    _publish(store, "old", text="{}", now=0.0)
    _publish(store, "fresh", text="{}", now=0.0)
    # Both are old, but "fresh" is running → never evicted.
    evicted = store.evict_expired(running_ids={"fresh"}, now=1000.0)
    assert evicted == ["old"]
    assert store.get("old") is None
    assert store.get("fresh") is not None
    assert not (store.root / "old").exists()


def test_eviction_leaves_tombstone_for_known_but_gone(tmp_path):
    store = ExtractionStore(root=tmp_path, ttl_s=100)
    _publish(store, "old", now=0.0)
    assert store.was_evicted("old") is False
    store.evict_expired(running_ids=set(), now=1000.0)
    # The id is gone from the index but remembered as evicted → routes answer 410.
    assert store.get("old") is None
    assert store.was_evicted("old") is True
    assert store.was_evicted("never-existed") is False


def test_tombstone_overflow_drops_oldest(tmp_path):
    # The bounded tombstone must drop the oldest id once it overflows maxlen,
    # keeping the deque and set in sync.
    import collections
    store = ExtractionStore(root=tmp_path, ttl_s=100)
    store._tombstones = collections.deque(maxlen=2)  # shrink the cap for the test
    for jid in ("a", "b", "c"):  # publish then evict each → 3 tombstones, cap 2
        _publish(store, jid, now=0.0)
    store.evict_expired(running_ids=set(), now=1000.0)
    assert store.was_evicted("c") and store.was_evicted("b")
    assert not store.was_evicted("a")  # oldest dropped past maxlen
    assert len(store._tombstone_set) == len(store._tombstones) == 2


def test_janitor_thread_can_be_joined(tmp_path, monkeypatch):
    # Regression: Janitor._stop must not shadow threading.Thread._stop, or join()
    # raises "Event object is not callable".
    monkeypatch.setenv("TRANSCRIPT_DATA_DIR", str(tmp_path / "store"))
    import transcript.server as server
    jan = server.Janitor(server.JobStore(), ExtractionStore(root=tmp_path / "s"),
                         interval_s=0.01)
    jan.start()
    jan.stop()
    jan.join(timeout=2.0)
    assert not jan.is_alive()


def test_eviction_skips_leased_bundle(tmp_path):
    store = ExtractionStore(root=tmp_path, ttl_s=100)
    _publish(store, "leased", now=0.0)
    with store.lease("leased") as job_dir:
        assert job_dir is not None
        evicted = store.evict_expired(running_ids=set(), now=1000.0)
        assert evicted == []  # leased → not evicted mid-stream


def test_startup_scan_rebuilds_index(tmp_path):
    store = ExtractionStore(root=tmp_path, ttl_s=100)
    _publish(store, "survivor", now=5.0)
    # New store over the same root → rebuilds from the on-disk manifest.
    store2 = ExtractionStore(root=tmp_path, ttl_s=100)
    assert store2.get("survivor") is not None
    assert store2.read_result("survivor", bump=False) == "{}"


def test_startup_recovers_valid_bundle_with_invalid_manifest(tmp_path):
    store = ExtractionStore(root=tmp_path, ttl_s=100)
    _publish(store, "survivor", text='{"kind":"image_note"}', now=5.0)
    _publish(store, "scalar", text='{"kind":"image_note"}', now=5.0)
    _publish(store, "bad-time", text='{"kind":"image_note"}', now=5.0)
    (store.root / "survivor" / "manifest.json").write_text("{")
    (store.root / "scalar" / "manifest.json").write_text("[]")
    (store.root / "bad-time" / "manifest.json").write_text(
        '{"id":"bad-time","kind":"image_note","status":"done",'
        '"created_at":5,"last_access":"later"}'
    )

    store2 = ExtractionStore(root=tmp_path, ttl_s=100)
    for job_id in ("survivor", "scalar", "bad-time"):
        assert store2.read_result(job_id, bump=False) == '{"kind":"image_note"}'
        assert store2.get(job_id)["kind"] == "image_note"
    assert (store2.root / "survivor" / "assets" / "card-000.jpg").is_file()


def test_startup_gcs_partial_publish(tmp_path):
    store = ExtractionStore(root=tmp_path, ttl_s=100)
    # A final dir with no result.json (a crash before/around the rename).
    partial = store.root / "partial"
    partial.mkdir()
    (partial / "manifest.json").write_text("{}")
    store2 = ExtractionStore(root=tmp_path, ttl_s=100)
    assert store2.get("partial") is None
    assert not partial.exists()


def test_record_rejects_unsafe_asset_keys(tmp_path):
    import pytest
    store = ExtractionStore(root=tmp_path, ttl_s=100)
    src = store.root / "_src.jpg"
    src.write_bytes(b"x")
    for bad in ("../escape.jpg", "/abs.jpg", "result.json", "manifest.json",
                "a/../b.jpg", "C:/x.jpg", "assets\\card.jpg", "",
                "assets//x.jpg", "assets/./x.jpg"):  # path aliases
        with pytest.raises(ValueError):
            store.record("j", "image_note", "{}", [(bad, src)])
    # Duplicate keys are rejected too — including case-only variants (a.jpg/A.jpg
    # collide on a case-insensitive FS).
    with pytest.raises(ValueError):
        store.record("j", "image_note", "{}", [("a.jpg", src), ("a.jpg", src)])
    with pytest.raises(ValueError):
        store.record("j", "image_note", "{}", [("assets/a.jpg", src), ("assets/A.jpg", src)])
    # A rejected key must not leave a partial staging dir behind.
    assert not (store.staging_dir("j")).exists()


def test_gc_staging_reaps_orphans_but_keeps_running(tmp_path):
    store = ExtractionStore(root=tmp_path, ttl_s=100)
    store.staging_dir("orphan").mkdir(parents=True)
    store.staging_dir("running").mkdir(parents=True)
    removed = store.gc_staging(running_ids={"running"})
    assert removed == 1
    assert not store.staging_dir("orphan").exists()
    assert store.staging_dir("running").exists()  # an in-flight publish is preserved


def test_startup_gcs_stale_staging(tmp_path):
    store = ExtractionStore(root=tmp_path, ttl_s=100)
    staging = store.staging_dir("crashed")
    staging.mkdir(parents=True)
    (staging / "result.json").write_text("{}")
    # Re-scan: stale staging dirs (no completed final) are GC'd.
    ExtractionStore(root=tmp_path, ttl_s=100)
    assert not staging.exists()
