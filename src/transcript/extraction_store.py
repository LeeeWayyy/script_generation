"""Durable on-disk store for completed extractions (plan §Interface, step 0b).

``JobStore`` is in-memory; on restart its ``text``/``assets[]``/``meta`` are gone
while the asset files would survive — an un-fetchable, leaking bundle. So a
completed extraction is persisted to disk:

    <root>/<id>/result.json     # immutable, hashed bundle member (the envelope)
    <root>/<id>/assets/...      # the asset files at their exact AssetRef.key
    <root>/<id>/manifest.json   # MUTABLE side manifest: created_at/last_access/...

The TTL timestamps live in ``manifest.json`` — NOT in ``result.json`` — so a
fetch bumping ``last_access`` never rewrites the immutable bundle bytes.

Publish ordering is atomic: everything is built under ``<root>/.staging/<id>/``
and ``os.rename``d into place; only then is the in-memory job marked ``done``.
A crash before the rename leaves a staging dir with no final ``result.json``; the
startup scan GCs those (and any final dir missing ``result.json``) so the durable
disk never leaks on crash.

This module is pure stdlib (json/os/threading/time) — no heavy deps, no FastAPI.
"""

from __future__ import annotations

import collections
import json
import os
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

DEFAULT_TTL_S = 7 * 24 * 3600  # a completed bundle lives a week since last access


def _fsync_dir(path: Path) -> None:
    """Best-effort fsync of a directory (persists renames/creates within it).
    No-op on platforms/filesystems that reject directory fds."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except (OSError, ValueError):
        pass


def _fsync_tree(root: Path) -> None:
    """fsync every regular file AND directory under ``root``, then ``root`` —
    so a power loss can't lose a freshly-written file whose containing subdir
    entry (e.g. ``assets/``) was not itself made durable."""
    for p in root.rglob("*"):
        if p.is_file():
            try:
                fd = os.open(str(p), os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except OSError:
                pass
        elif p.is_dir():
            _fsync_dir(p)
    _fsync_dir(root)


def default_root() -> Path:
    """Where durable extraction bundles live (override with $TRANSCRIPT_DATA_DIR)."""
    env = os.environ.get("TRANSCRIPT_DATA_DIR")
    base = Path(env) if env else Path.home() / ".cache" / "transcript" / "extractions"
    return base


class ExtractionStore:
    """Filesystem-backed index of completed extractions + a read-lease registry.

    Thread-safe. The janitor (see server) calls :meth:`evict_expired`; HTTP
    handlers call :meth:`record`, :meth:`get`, :meth:`lease`.
    """

    RESULT_NAME = "result.json"
    MANIFEST_NAME = "manifest.json"
    STAGING_DIR = ".staging"
    EVICTING_DIR = ".evicting"

    @classmethod
    def _validate_key(cls, key: str, seen: set[str]) -> None:
        """Enforce the server-side AssetRef.key invariant (plan §0): unique,
        POSIX-relative, non-empty, not ``result.json``, not absolute, no ``..``
        component — so duplicate members / path aliases can't make verification
        ambiguous or let a key escape the staging dir."""
        if not key or key.endswith("/"):
            raise ValueError(f"empty/invalid asset key: {key!r}")
        # Keys must be POSIX-relative: reject backslashes outright (validating on a
        # normalized copy while writing the raw key would otherwise let
        # "assets\\card.jpg" become a literal-backslash bundle member).
        if "\\" in key:
            raise ValueError(f"non-POSIX asset key (backslash) rejected: {key!r}")
        norm = key
        # Reject POSIX-absolute and Windows drive paths — absolute ("C:/x") AND
        # drive-relative ("C:x") — but not a stray colon in a POSIX name
        # ("0:00.jpg", whose first char isn't a letter).
        is_drive = len(norm) >= 2 and norm[0].isalpha() and norm[1] == ":"
        if norm.startswith("/") or is_drive:
            raise ValueError(f"absolute asset key rejected: {key!r}")
        parts = norm.split("/")
        if ".." in parts:
            raise ValueError(f"path-traversal asset key rejected: {key!r}")
        if norm in (cls.RESULT_NAME, cls.MANIFEST_NAME):
            raise ValueError(f"asset key collides with a reserved bundle name: {key!r}")
        if norm in seen:
            raise ValueError(f"duplicate asset key: {key!r}")
        seen.add(norm)

    def __init__(self, root: Optional[Path] = None, ttl_s: float = DEFAULT_TTL_S):
        self.root = Path(root) if root else default_root()
        self.ttl_s = ttl_s
        self._lock = threading.RLock()
        # job_id -> manifest dict (in-memory mirror of manifest.json)
        self._index: dict[str, dict] = {}
        # job_id -> active reader count (a non-zero lease blocks eviction)
        self._leases: dict[str, int] = {}
        # Bounded tombstone of recently-evicted ids so a client that knew the id
        # gets 410 (gone), not 404 (unknown), for the rest of the process life.
        # Capped so it can't grow without bound; cleared on restart (after which
        # a fully-evicted id is genuinely unknown → 404, the plan's documented
        # "unrecoverable after restart" alternative).
        self._tombstones: "collections.deque[str]" = collections.deque(maxlen=10_000)
        self._tombstone_set: set[str] = set()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / self.STAGING_DIR).mkdir(exist_ok=True)
        self._scan()

    # -- paths ---------------------------------------------------------------

    def _job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def staging_dir(self, job_id: str) -> Path:
        return self.root / self.STAGING_DIR / job_id

    def result_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / self.RESULT_NAME

    # -- startup scan / crash GC --------------------------------------------

    def _scan(self) -> None:
        """Rebuild the index from on-disk manifests; GC stale staging/partials."""
        with self._lock:
            staging_root = self.root / self.STAGING_DIR
            # Crash cleanup: any leftover staging or evicting dir has no completed
            # manifest and must be removed (else the durable disk leaks on crash).
            for transient in (staging_root, self.root / self.EVICTING_DIR):
                if transient.is_dir():
                    for child in transient.iterdir():
                        shutil.rmtree(child, ignore_errors=True)
            staging_root.mkdir(exist_ok=True)

            for child in self.root.iterdir():
                if not child.is_dir() or child.name in (self.STAGING_DIR, self.EVICTING_DIR):
                    continue
                result = child / self.RESULT_NAME
                manifest = child / self.MANIFEST_NAME
                if not result.is_file() or not manifest.is_file():
                    # A final dir missing its immutable result.json / manifest is a
                    # partial publish — unrecoverable, GC it.
                    shutil.rmtree(child, ignore_errors=True)
                    continue
                try:
                    self._index[child.name] = json.loads(manifest.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    shutil.rmtree(child, ignore_errors=True)

    # -- publish (atomic) ----------------------------------------------------

    def record(self, job_id: str, kind: str, result_json: str, assets: list[tuple[str, Path]],
               *, now: Optional[float] = None) -> None:
        """Atomically publish a completed extraction.

        ``assets`` is a list of ``(AssetRef.key, local_source_path)``. The bundle
        is built under the staging dir then ``os.rename``d into place; the caller
        marks the job ``done`` only after this returns.
        """
        now = time.time() if now is None else now
        # Validate every AssetRef.key up-front (before writing anything) so a bad
        # key can't escape staging or shadow result.json, and a partial staging
        # dir is never created on a rejected key.
        seen: set[str] = set()
        for key, _src in assets:
            self._validate_key(key, seen)

        staging = self.staging_dir(job_id)
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)

        # result.json — the immutable, hashed bundle member.
        (staging / self.RESULT_NAME).write_text(result_json, encoding="utf-8")
        # asset files at their exact AssetRef.key (POSIX-relative).
        for key, src in assets:
            dest = staging / key
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)

        manifest = {
            "id": job_id,
            "kind": kind,
            "status": "done",
            "created_at": now,
            "last_access": now,
        }
        (staging / self.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

        # Crash-durability: fsync the staged files + the staging dir BEFORE the
        # rename, so a power loss after status=done cannot expose a truncated
        # bundle (the rename is the publish point; an unsynced rename target could
        # otherwise survive while its contents are still in the page cache).
        _fsync_tree(staging)

        final = self._job_dir(job_id)
        with self._lock:
            if final.exists():
                shutil.rmtree(final, ignore_errors=True)
            os.rename(staging, final)
            _fsync_dir(final.parent)  # persist the rename itself
            self._index[job_id] = manifest

    # -- read ----------------------------------------------------------------

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            return self._index.get(job_id)

    def was_evicted(self, job_id: str) -> bool:
        """True if this id was a completed bundle we evicted this process life —
        so the routes can answer 410 (gone), not 404 (unknown)."""
        with self._lock:
            return job_id in self._tombstone_set

    def read_result(self, job_id: str, *, bump: bool = True) -> Optional[str]:
        """Return the immutable ``result.json`` text.

        When ``bump`` is set (the ``/result`` route), the read is performed UNDER
        a read-lease — so it shares the bundle's lease with ``/bundle``: a
        concurrent TTL sweep cannot evict-by-rename the dir mid-read, and
        ``last_access`` is bumped, so a client that fetched ``/result`` doesn't
        lose ``/bundle`` to TTL between the two calls. ``bump=False`` is a plain
        snapshot read used by the startup scan / tests."""
        if not bump:
            with self._lock:
                if job_id not in self._index:
                    return None
                path = self.result_path(job_id)
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                return None
        with self.lease(job_id) as job_dir:
            if job_dir is None:
                return None
            try:
                return (job_dir / self.RESULT_NAME).read_text(encoding="utf-8")
            except OSError:
                return None

    def _bump(self, job_id: str, *, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        manifest = self._index.get(job_id)
        if manifest is None:
            return
        manifest["last_access"] = now
        try:
            (self._job_dir(job_id) / self.MANIFEST_NAME).write_text(
                json.dumps(manifest), encoding="utf-8"
            )
        except OSError:
            pass

    @contextmanager
    def lease(self, job_id: str):
        """Hold a read-lease for the duration of a ``/bundle`` stream so a
        concurrent sweep cannot unlink the dir mid-stream. Yields the job dir, or
        ``None`` if the bundle is gone (TTL-evicted / lost on restart)."""
        with self._lock:
            available = job_id in self._index and self.result_path(job_id).is_file()
            if available:
                self._leases[job_id] = self._leases.get(job_id, 0) + 1
                self._bump(job_id)
                job_dir = self._job_dir(job_id)
        # Yield OUTSIDE the lock in both branches so a caller that does real work
        # in the None-path can't block all store operations.
        if not available:
            yield None
            return
        try:
            yield job_dir
        finally:
            with self._lock:
                n = self._leases.get(job_id, 0) - 1
                if n <= 0:
                    self._leases.pop(job_id, None)
                else:
                    self._leases[job_id] = n

    # -- eviction ------------------------------------------------------------

    def gc_staging(self, *, running_ids: set[str]) -> int:
        """Remove orphaned staging dirs (a failed ``record`` leaves one behind).

        A staging dir for a job that is NOT currently running can never be written
        again, so it is safe to delete mid-session rather than waiting for the next
        startup scan. Returns the number removed."""
        staging_root = self.root / self.STAGING_DIR
        if not staging_root.is_dir():
            return 0
        # Pick targets under the lock, then rmtree OUTSIDE it (blocking disk I/O
        # under the lock would stall get/lease/record) — mirrors evict_expired.
        with self._lock:
            targets = [c for c in staging_root.iterdir() if c.name not in running_ids]
        for child in targets:
            shutil.rmtree(child, ignore_errors=True)
        return len(targets)

    def evict_expired(self, *, running_ids: set[str], now: Optional[float] = None) -> list[str]:
        """Evict bundles whose TTL lapsed. Never evicts a running job or a leased
        (mid-stream) one. Uses evict-by-rename-then-delete so an in-flight reader
        keeps a valid dir handle even as the sweep removes the published path."""
        now = time.time() if now is None else now
        evicting_root = self.root / self.EVICTING_DIR
        evicting_root.mkdir(exist_ok=True)
        tombstones: list[tuple[str, Path]] = []
        with self._lock:
            candidates = [
                jid for jid, m in self._index.items()
                if jid not in running_ids
                and jid not in self._leases
                and (now - m.get("last_access", 0)) > self.ttl_s
            ]
            for jid in candidates:
                src = self._job_dir(jid)
                tomb = evicting_root / jid
                try:
                    os.rename(src, tomb)
                except OSError:
                    continue
                self._index.pop(jid, None)
                if jid not in self._tombstone_set:
                    if len(self._tombstones) == self._tombstones.maxlen:
                        self._tombstone_set.discard(self._tombstones[0])
                    self._tombstones.append(jid)
                    self._tombstone_set.add(jid)
                tombstones.append((jid, tomb))
        # Delete tombstones outside the lock.
        evicted_ids: list[str] = []
        for jid, tomb in tombstones:
            shutil.rmtree(tomb, ignore_errors=True)
            evicted_ids.append(jid)
        return evicted_ids
