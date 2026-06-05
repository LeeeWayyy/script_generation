"""Mac-side client for the extraction API — ``extract-remote`` (plan §Interface).

Needs only ``requests`` + stdlib ``zipfile``. Submits a URL / file / manual-export
bundle to a transcript-server, polls the **extraction** routes (never the legacy
``/jobs/*`` API), downloads the one zip bundle, unzips it, and **verifies each
``AssetRef``'s ``sha256``/``size``** while **rejecting absolute / ``..`` members**
before writing — so a compromised/misconfigured server can't write outside
``--out-dir``.

No ``--kind audio`` alias: ``audio_extraction`` is spelled out so "audio" never
names two commands that hash different bytes (plain ASR stays ``transcript-remote``).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path

from ._remote_http import build_headers, poll_until_done, stderr_note

KINDS = ("video", "audio_extraction", "image_note")
# Note: plain audio extensions are deliberately NOT mapped. `audio_extraction`
# is podcast-only (it requires RSS/enclosure provenance), so a bare audio file/URL
# must not auto-route here — plain ASR stays on `transcript-remote`. A user who
# really wants a podcast envelope passes --kind audio_extraction + --feed-url/
# --enclosure-url explicitly.
_EXT_TO_KIND = {
    ".zip": "image_note", ".tar": "image_note", ".gz": "image_note", ".tgz": "image_note",
    ".mp4": "video", ".mkv": "video", ".mov": "video", ".webm": "video", ".avi": "video",
}
_AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac"}


def sniff_kind(source: str) -> str:
    """Sniff modality by extension; hard-fail on ambiguity (explicit --kind wins)."""
    suffix = Path(source.split("?")[0]).suffix.lower()
    kind = _EXT_TO_KIND.get(suffix)
    if kind is None:
        if suffix in _AUDIO_EXTS:
            raise SystemExit(
                f"{source!r} looks like plain audio. Use `transcript-remote` for "
                f"plain ASR, or pass `--kind audio_extraction` with --feed-url / "
                f"--enclosure-url for a podcast envelope."
            )
        raise SystemExit(
            f"Could not infer --kind from {source!r}; pass --kind "
            f"{{{','.join(KINDS)}}} explicitly."
        )
    return kind


class BundleVerificationError(Exception):
    pass


_CHUNK = 1024 * 1024


def _check_key(key: str, what: str, *, reserved: tuple[str, ...] = ()) -> str:
    """Reject absolute / drive-letter / ``..`` / backslash / reserved paths."""
    kp = str(key)
    if not kp or kp.endswith("/"):
        raise BundleVerificationError(f"empty/invalid {what}: {key!r}")
    if "\\" in kp:  # keys/members are POSIX-relative; a backslash is never valid
        raise BundleVerificationError(f"non-POSIX {what} rejected: {key!r}")
    is_drive = len(kp) >= 2 and kp[0].isalpha() and kp[1] == ":"  # "C:/x" AND "C:x"
    if kp.startswith("/") or Path(kp).is_absolute() or is_drive or ".." in kp.split("/"):
        raise BundleVerificationError(f"unsafe {what} rejected: {key!r}")
    if kp in reserved:
        raise BundleVerificationError(f"{what} collides with a reserved name: {key!r}")
    return kp


def _safe_member_names(zf: zipfile.ZipFile) -> list[str]:
    """Return the safe regular-file member names; reject unsafe paths/symlinks and
    duplicate names (a duplicate makes verification ambiguous)."""
    names: list[str] = []
    seen: set[str] = set()
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = _check_key(info.filename, "bundle member")
        if ((info.external_attr >> 16) & 0o170000) == 0o120000:
            raise BundleVerificationError(f"symlink bundle member rejected: {name!r}")
        if name in seen:
            raise BundleVerificationError(f"duplicate bundle member rejected: {name!r}")
        seen.add(name)
        names.append(name)
    return names


def _stream_member_hash(zf: zipfile.ZipFile, name: str) -> tuple[str, int]:
    """sha256 + size of a zip member, read in bounded chunks (never whole-in-RAM)."""
    hasher = hashlib.sha256()
    size = 0
    with zf.open(name) as m:
        for chunk in iter(lambda: m.read(_CHUNK), b""):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def unpack_and_verify(zip_source, out_dir: Path) -> dict:
    """Verify a bundle (streaming, bounded RAM), then write only verified members.

    ``zip_source`` is a path to the downloaded zip on disk (preferred) or raw
    bytes (for small/in-test bundles). The member set must be EXACTLY
    ``{"result.json"} ∪ {asset.key}`` — no extras/duplicates, and ``result.json``
    is reserved (an asset can't claim it) — and every asset's ``sha256``/``size``
    must match BEFORE anything is written to ``out_dir``. Assets are hashed and
    copied in 1 MiB chunks, so a large (valid) bundle never lands wholly in RAM.
    Returns the parsed ``result.json`` envelope.
    """
    def _open():
        if isinstance(zip_source, (bytes, bytearray)):
            return zipfile.ZipFile(io.BytesIO(zip_source))
        return zipfile.ZipFile(zip_source)

    with _open() as zf:
        names = set(_safe_member_names(zf))
        if "result.json" not in names:
            raise BundleVerificationError("bundle is missing result.json")
        envelope = json.loads(zf.read("result.json").decode("utf-8"))

        asset_keys = [_check_key(a["key"], "asset key", reserved=("result.json",))
                      for a in envelope.get("assets", [])]
        if len(asset_keys) != len(set(asset_keys)):
            raise BundleVerificationError("duplicate asset key in envelope")
        extra = names - {"result.json", *asset_keys}
        if extra:
            raise BundleVerificationError(f"bundle has unexpected members: {sorted(extra)}")

        # Pass 1 — verify every asset (streaming hash), writing NOTHING yet.
        for asset, key in zip(envelope.get("assets", []), asset_keys):
            if key not in names:
                raise BundleVerificationError(f"asset listed but missing from bundle: {key}")
            digest, size = _stream_member_hash(zf, key)
            if size != asset["size"]:
                raise BundleVerificationError(
                    f"asset size mismatch for {key}: {size} != {asset['size']}"
                )
            if digest != asset["sha256"]:
                raise BundleVerificationError(f"asset sha256 mismatch for {key}")

        # Pass 2 — everything verified; stream each member to disk.
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("result.json", *asset_keys):
            dest = out_dir / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as m, dest.open("wb") as o:
                for chunk in iter(lambda: m.read(_CHUNK), b""):
                    o.write(chunk)
    return envelope


def main(argv: list[str] | None = None) -> int:
    import requests

    p = argparse.ArgumentParser(
        prog="extract-remote",
        description="Submit a URL/file/bundle to a transcript-server extraction "
        "route and fetch the verified bundle.",
    )
    p.add_argument("source", nargs="?",
                   help="URL, local media file, or manual-export bundle (zip/tar). For "
                   "audio_extraction a positional feed URL is also accepted, or omit it "
                   "and pass --feed-url / --enclosure-url.")
    p.add_argument("--kind", choices=KINDS, help="Modality (else inferred from the extension).")
    p.add_argument("--frames", action="store_true", help="Extract video frames (with --kind video).")
    p.add_argument("--cadence", type=float, help="Frame cadence in seconds (with --frames).")
    p.add_argument("--feed-url", help="RSS feed URL (audio_extraction).")
    p.add_argument("--episode-guid", help="Episode GUID selector (audio_extraction).")
    p.add_argument("--episode-url", help="Episode page URL selector (audio_extraction).")
    p.add_argument("--episode-title", help="Episode title selector (last-resort, audio_extraction).")
    p.add_argument("--episode-published", help="Episode published date selector (audio_extraction).")
    p.add_argument("--enclosure-url", help="Explicit user-supplied enclosure URL (audio_extraction).")
    p.add_argument("--out-dir", default=".", help="Where to unpack the verified bundle.")
    p.add_argument("--server", default=os.environ.get("TRANSCRIPT_SERVER", "http://localhost:8000"))
    p.add_argument("--token", default=os.environ.get("TRANSCRIPT_TOKEN"))
    p.add_argument("--no-diarize", dest="diarize", action="store_false", default=True)
    p.add_argument("--language")
    p.add_argument("--poll", type=float, default=3.0)
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args(argv)

    base = args.server.rstrip("/")
    headers = build_headers(args.token)
    note = stderr_note(args.quiet)

    if not args.kind and not args.source:
        print("Error: pass a source, or --kind with the relevant flags.", file=sys.stderr)
        return 1
    kind = args.kind or sniff_kind(args.source)

    feed_url, enclosure_url = args.feed_url, args.enclosure_url
    # For audio_extraction (podcast-only), a positional URL source is the feed URL
    # (the server ignores `url` for this kind) — route it so the documented
    # `extract-remote --kind audio_extraction <feed-url> ...` form works.
    if kind == "audio_extraction" and args.source \
            and args.source.startswith(("http://", "https://")) \
            and not feed_url and not enclosure_url:
        feed_url = args.source
    if kind == "audio_extraction" and not feed_url and not enclosure_url:
        print("Error: audio_extraction needs --feed-url (+selector) or --enclosure-url "
              "(or a positional feed URL).", file=sys.stderr)
        return 1

    data = {"kind": kind, "diarize": str(args.diarize).lower()}
    if args.frames:
        data["frames"] = "true"
    if args.cadence is not None:
        data["cadence_s"] = str(args.cadence)
    for k, v in (("feed_url", feed_url), ("episode_guid", args.episode_guid),
                 ("episode_url", args.episode_url), ("episode_title", args.episode_title),
                 ("episode_published", args.episode_published),
                 ("enclosure_url", enclosure_url), ("language", args.language)):
        if v:
            data[k] = v

    files = None
    fh = None
    # audio_extraction submits no url/file (provenance comes from feed/enclosure).
    submit_source = args.source if kind != "audio_extraction" else None
    is_url = bool(submit_source) and submit_source.startswith(("http://", "https://"))
    try:
        if kind == "audio_extraction":
            note(f"Submitting {kind} (feed/enclosure) to {base} ...")
        elif is_url:
            data["url"] = submit_source
            note(f"Submitting {kind} URL to {base} ...")
        elif submit_source:
            path = Path(submit_source).expanduser()
            if not path.is_file():
                print(f"Error: file not found: {path}", file=sys.stderr)
                return 1
            fh = path.open("rb")
            files = {"file": (path.name, fh)}
            note(f"Uploading {path.name} ({kind}) to {base} ...")
        else:
            print(f"Error: kind={kind} requires a URL or file source.", file=sys.stderr)
            return 1
        try:
            r = requests.post(f"{base}/extractions", data=data, files=files, headers=headers)
        finally:
            if fh:
                fh.close()
    except requests.RequestException as exc:
        print(f"Error: could not reach server at {base}: {exc}", file=sys.stderr)
        return 1

    if r.status_code == 401:
        print("Error: unauthorized. Set --token / $TRANSCRIPT_TOKEN.", file=sys.stderr)
        return 1
    if not r.ok:
        print(f"Error: server rejected job ({r.status_code}): {r.text}", file=sys.stderr)
        return 1

    job_id = r.json()["id"]
    note(f"Extraction {job_id} queued. Polling ...")
    try:
        poll_until_done(requests, f"{base}/extractions/{job_id}", headers,
                        poll=args.poll, timeout=args.timeout, note=note)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Stream the bundle to a temp file (a connect+read timeout so a stalled server
    # can't hang the client forever; the body is never held wholly in RAM).
    import tempfile
    tmp = tempfile.NamedTemporaryFile(prefix="bundle-", suffix=".zip", delete=False)
    try:
        with requests.get(f"{base}/extractions/{job_id}/bundle", headers=headers,
                          stream=True, timeout=(30, args.timeout)) as rr:
            if rr.status_code in (404, 410):
                print(f"Error: bundle unavailable ({rr.status_code})", file=sys.stderr)
                return 1
            if not rr.ok:
                print(f"Error fetching bundle ({rr.status_code})", file=sys.stderr)
                return 1
            for chunk in rr.iter_content(chunk_size=1024 * 1024):
                tmp.write(chunk)
        tmp.close()

        out_dir = Path(args.out_dir).expanduser() / job_id
        try:
            envelope = unpack_and_verify(Path(tmp.name), out_dir)
        except BundleVerificationError as exc:
            print(f"Error: bundle verification failed: {exc}", file=sys.stderr)
            return 1
    except requests.RequestException as exc:
        print(f"Error fetching bundle: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    note(f"Verified bundle -> {out_dir} ({len(envelope.get('assets', []))} assets)")
    sys.stdout.write(envelope.get("text", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
