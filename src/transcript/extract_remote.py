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
import math
import os
import shutil
import sys
import tempfile
import unicodedata
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

from ._remote_http import (
    build_headers,
    poll_until_done,
    request_timeout,
    response_job_id,
    stderr_note,
    validate_common_options,
)
from .extraction import KINDS
from .frames import MIN_CADENCE_S
from .ingest import is_url
from .types import has_windows_drive_prefix, is_windows_reserved_basename
from .zip_safety import preflight_zip

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
    suffix = Path(urlsplit(source).path if is_url(source) else source).suffix.lower()
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
_MAX_RESULT_JSON_BYTES = 64 * 1024 * 1024  # the envelope is text; 64 MiB is enormous
_MAX_TOTAL_ASSET_BYTES = 2 * 1024 * 1024 * 1024
_MAX_MEMBERS = 10_000
_MAX_CENTRAL_DIRECTORY_BYTES = 64 * 1024 * 1024
# ponytail: mirrors the server's archive ceiling; expose a flag if legitimate
# video-frame bundles need more than 2 GiB of uncompressed assets.
_MAX_BUNDLE_BYTES = _MAX_TOTAL_ASSET_BYTES + 128 * 1024 * 1024


def _path_token(path: str) -> str:
    """Filesystem collision token for case-insensitive, normalizing clients."""
    return unicodedata.normalize("NFC", path).casefold()


def _check_key(key: str, what: str, *, reserved: tuple[str, ...] = ()) -> str:
    """Reject absolute / drive-letter / ``..`` / backslash / reserved paths."""
    kp = str(key)
    if not kp or kp.endswith("/"):
        raise BundleVerificationError(f"empty/invalid {what}: {key!r}")
    if "\\" in kp:  # keys/members are POSIX-relative; a backslash is never valid
        raise BundleVerificationError(f"non-POSIX {what} rejected: {key!r}")
    is_drive = has_windows_drive_prefix(kp)  # "C:/x" AND "C:x"
    parts = kp.split("/")
    if (kp.startswith("/") or Path(kp).is_absolute() or is_drive or ".." in parts
            or "" in parts or "." in parts):  # incl. "a//b" / "a/./b" aliases
        raise BundleVerificationError(f"unsafe {what} rejected: {key!r}")
    for part in parts:
        if (part.endswith((" ", "."))
                or any(char in '<>:"|?*' or ord(char) < 32 or ord(char) == 127
                       for char in part)
                or is_windows_reserved_basename(part)):
            raise BundleVerificationError(f"Windows-unsafe {what} rejected: {key!r}")
    if _path_token(kp) in {_path_token(name) for name in reserved}:
        raise BundleVerificationError(f"{what} collides with a reserved name: {key!r}")
    return kp


def _safe_member_names(zf: zipfile.ZipFile) -> list[str]:
    """Return the safe regular-file member names; reject unsafe paths/symlinks and
    duplicate names (a duplicate makes verification ambiguous)."""
    names: list[str] = []
    seen: set[str] = set()
    for info in zf.infolist():
        if info.flag_bits & 0x1:
            raise BundleVerificationError(
                f"encrypted bundle member rejected: {info.filename!r}"
            )
        if ((info.external_attr >> 16) & 0o170000) == 0o120000:
            raise BundleVerificationError(f"symlink bundle member rejected: {info.filename!r}")
        if info.is_dir():
            # Directory entries are not part of the payload, but their path
            # components still need the same cross-platform validation.
            _check_key(info.filename.rstrip("/"), "bundle member")
            continue
        name = _check_key(info.filename, "bundle member")
        # Dedup case-insensitively — on macOS/Windows two members differing only by
        # case write the same file, silently overwriting one.
        token = _path_token(name)
        if token in seen:
            raise BundleVerificationError(f"duplicate bundle member rejected: {name!r}")
        seen.add(token)
        names.append(name)
    return names


def unpack_and_verify(zip_source, out_dir: Path) -> dict:
    """Stream a verified bundle into staging, then atomically publish it.

    ``zip_source`` is a path to the downloaded zip on disk (preferred) or raw
    bytes (for small/in-test bundles). The member set must be EXACTLY
    ``{"result.json"} ∪ {asset.key}`` — no extras/duplicates, and ``result.json``
    is reserved (an asset can't claim it) — and every asset's ``sha256``/``size``
    must match before the sibling staging directory is renamed to ``out_dir``.
    Assets are hashed while copied in 1 MiB chunks, so each is streamed once and
    a large valid bundle never lands wholly in RAM.
    Returns the parsed ``result.json`` envelope.
    """
    def _open():
        try:
            if isinstance(zip_source, (bytes, bytearray)):
                source = io.BytesIO(zip_source)
                preflight_zip(
                    source,
                    max_entries=_MAX_MEMBERS,
                    max_central_directory_bytes=_MAX_CENTRAL_DIRECTORY_BYTES,
                )
                source.seek(0)
                return zipfile.ZipFile(source)
            with Path(zip_source).open("rb") as source:
                preflight_zip(
                    source,
                    max_entries=_MAX_MEMBERS,
                    max_central_directory_bytes=_MAX_CENTRAL_DIRECTORY_BYTES,
                )
            return zipfile.ZipFile(zip_source)
        except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
            raise BundleVerificationError(f"invalid zip bundle: {exc}") from exc

    with _open() as zf:
        names = set(_safe_member_names(zf))
        if "result.json" not in names:
            raise BundleVerificationError("bundle is missing result.json")
        # Cap result.json before reading it into memory (it's the one member we
        # must fully parse) so a hostile server can't OOM the client with a huge
        # envelope, and tolerate a malformed envelope as a verification failure.
        if zf.getinfo("result.json").file_size > _MAX_RESULT_JSON_BYTES:
            raise BundleVerificationError("result.json is implausibly large")
        try:
            result_json = zf.read("result.json")
            envelope = json.loads(result_json.decode("utf-8"))
            if not isinstance(envelope, dict):
                raise TypeError("envelope must be an object")
            if envelope.get("kind") not in KINDS:
                raise ValueError("invalid extraction kind")
            if not isinstance(envelope.get("text"), str):
                raise TypeError("text must be a string")
            assets = envelope.get("assets")
            if not isinstance(assets, list):
                raise TypeError("assets must be a list")
            asset_keys = []
            asset_meta = []
            for asset in assets:
                if not isinstance(asset, dict) or not isinstance(asset.get("key"), str):
                    raise TypeError("asset/key has the wrong type")
                key = _check_key(asset["key"], "asset key", reserved=("result.json",))
                sha = asset.get("sha256")
                if (not isinstance(sha, str) or len(sha) != 64
                        or any(c not in "0123456789abcdefABCDEF" for c in sha)):
                    raise ValueError(f"invalid asset sha256 for {key}")
                size = asset.get("size")
                if type(size) is not int:  # bool is not a valid JSON byte count
                    raise TypeError(f"invalid asset size for {key}")
                if size < 0:
                    raise ValueError(f"asset size cannot be negative for {key}")
                if not isinstance(asset.get("media_type"), str):
                    raise TypeError(f"invalid asset media_type for {key}")
                asset_keys.append(key)
                asset_meta.append((sha.lower(), size))
        except (ValueError, KeyError, TypeError, AttributeError, OSError, RuntimeError,
                zipfile.BadZipFile) as exc:
            raise BundleVerificationError(f"malformed result.json envelope: {exc}") from exc
        # Dedup case-insensitively (matches the server + a case-insensitive client FS).
        if len({_path_token(k) for k in asset_keys}) != len(asset_keys):
            raise BundleVerificationError("duplicate asset key in envelope")
        if sum(size for _, size in asset_meta) > _MAX_TOTAL_ASSET_BYTES:
            raise BundleVerificationError("bundle assets exceed the 2 GiB safety limit")
        extra = {_path_token(n) for n in names} - {
            _path_token("result.json"), *(_path_token(k) for k in asset_keys),
        }
        if extra:
            raise BundleVerificationError("bundle has unexpected members")
        for key, (_, want_size) in zip(asset_keys, asset_meta):
            if key not in names:
                raise BundleVerificationError(f"asset listed but missing from bundle: {key}")
            if zf.getinfo(key).file_size != want_size:
                raise BundleVerificationError(
                    f"asset size mismatch for {key}: "
                    f"{zf.getinfo(key).file_size} != {want_size}"
                )

        # Unpack each asset exactly once while hashing. The sibling staging dir is
        # cleaned on every failure and published with one atomic rename only after
        # every member verifies.
        out_dir = Path(out_dir)
        staging = None
        try:
            out_dir.parent.mkdir(parents=True, exist_ok=True)
            if out_dir.exists():
                raise BundleVerificationError(f"destination already exists: {out_dir}")
            staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}-", dir=out_dir.parent))
            (staging / "result.json").write_bytes(result_json)
            for key, (want_sha, want_size) in zip(asset_keys, asset_meta):
                dest = staging / key
                dest.parent.mkdir(parents=True, exist_ok=True)
                hasher = hashlib.sha256()
                size = 0
                with zf.open(key) as m, dest.open("wb") as o:
                    for chunk in iter(lambda: m.read(_CHUNK), b""):
                        hasher.update(chunk)
                        size += len(chunk)
                        if size > want_size:
                            raise BundleVerificationError(
                                f"asset size exceeds declared size for {key}"
                            )
                        o.write(chunk)
                if size != want_size:
                    raise BundleVerificationError(
                        f"asset size mismatch for {key}: {size} != {want_size}"
                    )
                if hasher.hexdigest() != want_sha:
                    raise BundleVerificationError(f"asset sha256 mismatch for {key}")
            staging.rename(out_dir)
            staging = None
        except BundleVerificationError:
            raise
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            raise BundleVerificationError(f"failed writing bundle to disk: {exc}") from exc
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)
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
    p.add_argument("--min-speakers", type=int)
    p.add_argument("--max-speakers", type=int)
    p.add_argument("--detect-music", action="store_true", help="Opt in to music tagging.")
    p.add_argument("--poll", type=float, default=3.0)
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args(argv)

    option_error = validate_common_options(
        poll=args.poll,
        timeout=args.timeout,
        diarize=args.diarize,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )
    if option_error:
        print(f"Error: {option_error}.", file=sys.stderr)
        return 1

    base = args.server.rstrip("/")
    headers = build_headers(args.token)
    note = stderr_note(args.quiet)

    if not args.kind and not args.source:
        print("Error: pass a source, or --kind with the relevant flags.", file=sys.stderr)
        return 1
    kind = args.kind or sniff_kind(args.source)

    podcast_values = (
        args.feed_url,
        args.episode_guid,
        args.episode_url,
        args.episode_title,
        args.episode_published,
        args.enclosure_url,
    )
    if kind != "audio_extraction" and any(podcast_values):
        print(
            "Error: podcast selector options require --kind audio_extraction.",
            file=sys.stderr,
        )
        return 1
    if kind != "video" and (args.frames or args.cadence is not None):
        print("Error: --frames/--cadence require --kind video.", file=sys.stderr)
        return 1
    if args.cadence is not None and not args.frames:
        print("Error: --cadence requires --frames.", file=sys.stderr)
        return 1
    if args.cadence is not None and (
        not math.isfinite(args.cadence) or args.cadence < MIN_CADENCE_S
    ):
        print(f"Error: --cadence must be at least {MIN_CADENCE_S} seconds.", file=sys.stderr)
        return 1
    if kind == "image_note" and args.source and is_url(args.source):
        print("Error: image_note requires a local archive upload.", file=sys.stderr)
        return 1
    if kind == "image_note" and (
        not args.diarize
        or args.language is not None
        or args.min_speakers is not None
        or args.max_speakers is not None
        or args.detect_music
    ):
        print(
            "Error: ASR/diarization/music options do not apply to image_note.",
            file=sys.stderr,
        )
        return 1

    feed_url, enclosure_url = args.feed_url, args.enclosure_url
    # For audio_extraction (podcast-only), a positional URL source is the feed URL
    # (the server ignores `url` for this kind) — route it so the documented
    # `extract-remote --kind audio_extraction <feed-url> ...` form works.
    if kind == "audio_extraction" and args.source:
        if not is_url(args.source):
            print(
                "Error: audio_extraction accepts a feed URL, not a local audio file.",
                file=sys.stderr,
            )
            return 1
        if feed_url or enclosure_url:
            print(
                "Error: positional feed URL conflicts with --feed-url/--enclosure-url.",
                file=sys.stderr,
            )
            return 1
        feed_url = args.source
    if feed_url and enclosure_url:
        print("Error: --feed-url and --enclosure-url are mutually exclusive.", file=sys.stderr)
        return 1
    selectors = (
        args.episode_guid,
        args.episode_url,
        args.episode_title,
        args.episode_published,
    )
    if enclosure_url and any(selectors):
        print("Error: episode selectors cannot be used with --enclosure-url.", file=sys.stderr)
        return 1
    if bool(args.episode_title) != bool(args.episode_published):
        print("Error: --episode-title and --episode-published must be used together.",
              file=sys.stderr)
        return 1
    fallback_selector = bool(args.episode_title and args.episode_published)
    if fallback_selector and (args.episode_guid or args.episode_url):
        print(
            "Error: title/published fallback cannot be combined with GUID or episode URL.",
            file=sys.stderr,
        )
        return 1
    if feed_url and not (args.episode_guid or args.episode_url or fallback_selector):
        print(
            "Error: --feed-url needs --episode-guid, "
            "--episode-url, or --episode-title plus --episode-published.",
            file=sys.stderr,
        )
        return 1
    if kind == "audio_extraction" and not feed_url and not enclosure_url:
        print("Error: audio_extraction needs --feed-url (+selector) or --enclosure-url "
              "(or a positional feed URL).", file=sys.stderr)
        return 1

    data = {
        "kind": kind,
        "diarize": str(args.diarize).lower(),
        "detect_music": str(args.detect_music).lower(),
    }
    if args.frames:
        data["frames"] = "true"
    if args.cadence is not None:
        data["cadence_s"] = str(args.cadence)
    if args.min_speakers is not None:
        data["min_speakers"] = str(args.min_speakers)
    if args.max_speakers is not None:
        data["max_speakers"] = str(args.max_speakers)
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
    source_is_url = bool(submit_source) and is_url(submit_source)
    try:
        if kind == "audio_extraction":
            note(f"Submitting {kind} (feed/enclosure) to {base} ...")
        elif source_is_url:
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
            r = requests.post(
                f"{base}/extractions",
                data=data,
                files=files,
                headers=headers,
                timeout=request_timeout(args.timeout),
            )
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

    try:
        job_id = response_job_id(r)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    note(f"Extraction {job_id} queued. Polling ...")
    try:
        poll_until_done(requests, f"{base}/extractions/{job_id}", headers,
                        poll=args.poll, timeout=args.timeout, note=note)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Stream the bundle to a temp file (a connect+read timeout so a stalled server
    # can't hang the client forever; the body is never held wholly in RAM).
    tmp = tempfile.NamedTemporaryFile(prefix="bundle-", suffix=".zip", delete=False)
    try:
        with requests.get(f"{base}/extractions/{job_id}/bundle", headers=headers,
                          stream=True, timeout=request_timeout(args.timeout)) as rr:
            if rr.status_code in (404, 410):
                print(f"Error: bundle unavailable ({rr.status_code})", file=sys.stderr)
                return 1
            if not rr.ok:
                print(f"Error fetching bundle ({rr.status_code})", file=sys.stderr)
                return 1
            try:
                content_length = int(rr.headers.get("Content-Length", 0))
            except (TypeError, ValueError):
                content_length = 0
            if content_length > _MAX_BUNDLE_BYTES:
                print("Error: bundle exceeds the client safety limit", file=sys.stderr)
                return 1
            downloaded = 0
            for chunk in rr.iter_content(chunk_size=1024 * 1024):
                downloaded += len(chunk)
                if downloaded > _MAX_BUNDLE_BYTES:
                    print("Error: bundle exceeds the client safety limit", file=sys.stderr)
                    return 1
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
            tmp.close()
        except OSError:
            pass
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    note(f"Verified bundle -> {out_dir} ({len(envelope.get('assets', []))} assets)")
    sys.stdout.write(envelope.get("text", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
