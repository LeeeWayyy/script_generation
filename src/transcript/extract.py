"""Per-kind extraction orchestration: media → :class:`ExtractionResult` + assets.

This is the server-side entry point the worker dispatches to. Each kind produces
an :class:`ExtractionResult` (text + modality lists + a provenance recipe) plus a
list of ``(AssetRef.key, local_path)`` asset files for the worker to publish into
the durable bundle. The ``job_id``/``server_version`` meta-stamp is added by the
worker (on ``ExtractionResult.meta``, never ``Transcript.meta``).

Heavy deps (OCR, ffmpeg, feedparser, whisperx) are reached only through the
already-lazy modality modules — nothing heavy is imported at module top.

Provenance fields are tagged recipe (stable) vs observation (varies) under
``meta["_provenance"]`` so the consumer hashes the returned artifact and uses the
recipe only to *explain* why a re-run differs.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
from pathlib import Path
from typing import Optional

from .extraction import AssetRef, Card, ExtractionResult, Frame, render_audio_text, \
    render_image_note_text
from .types import Transcript

log = logging.getLogger("transcript.extract")


def _asset_ref(key: str, path: Path) -> AssetRef:
    # Stream the hash in chunks — an asset can be large (frames/images up to the
    # archive cap), so never read the whole file into memory just for integrity.
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
            size += len(chunk)
    media_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
    return AssetRef(key=key, sha256=hasher.hexdigest(), size=size, media_type=media_type)


def _tag_provenance(meta: dict, *, recipe: list[str], observation: list[str]) -> None:
    meta["_provenance"] = {"recipe": recipe, "observation": observation}


# ---------------------------------------------------------------------------
# image_note — OCR'd cards from a manual-export archive
# ---------------------------------------------------------------------------


def extract_image_note(archive_path: Path, asset_dir: Path, *, ocr_engine=None
                       ) -> tuple[ExtractionResult, list[tuple[str, Path]]]:
    from .archive import extract_images
    from .ocr import OCR_PARAMS, run_ocr

    members = extract_images(archive_path, asset_dir / "_members")
    cards: list[Card] = []
    assets: list[AssetRef] = []
    asset_files: list[tuple[str, Path]] = []

    for idx, member in enumerate(members):
        key = f"assets/card-{idx:03d}{Path(member.basename).suffix.lower()}"
        try:
            ocr = run_ocr(member.path, engine=ocr_engine)
        except Exception as exc:  # noqa: BLE001 — record empty OCR, keep the card
            log.warning("OCR failed for %s: %s", member.basename, exc)
            ocr = None
        cards.append(Card(
            index=idx,
            ocr_text=ocr.ocr_text if ocr else "",
            image_ref=key,
            source_filename=member.basename,
            image_sha256=member.sha256,
            width=ocr.width if ocr else None,
            height=ocr.height if ocr else None,
            confidence=ocr.confidence if ocr else None,
            source_member=member.source_member,
            blocks=ocr.blocks if ocr else None,
        ))
        assets.append(_asset_ref(key, member.path))
        asset_files.append((key, member.path))

    import os
    meta: dict = {
        "ocr_engine": _ocr_engine_version(),
        # A model identifier (not just the raw lang, which ocr_params already has).
        "ocr_model": f"paddleocr-{OCR_PARAMS['lang']}",
        # The pinned weights dir, if any — two hosts with different weights at
        # different paths produce different ocr_text under identical params, so
        # the recipe records which weights were used to explain divergence.
        "ocr_model_dir": os.environ.get("TRANSCRIPT_OCR_MODEL_DIR"),
        "ocr_params": OCR_PARAMS,
    }
    _tag_provenance(meta, recipe=["ocr_engine", "ocr_model", "ocr_model_dir", "ocr_params"],
                    observation=["cards"])
    result = ExtractionResult(
        kind="image_note",
        text=render_image_note_text(cards),
        language=None,
        cards=cards,
        assets=assets,
        meta=meta,
    )
    return result, asset_files


def _ocr_engine_version() -> Optional[str]:
    from .engine import _pkg_version
    v = _pkg_version("paddleocr")
    return f"paddleocr@{v}" if v else None


# ---------------------------------------------------------------------------
# audio_extraction — podcast/RSS-resolved audio (ASR + RSS provenance)
# ---------------------------------------------------------------------------


def extract_audio_extraction(
    *, feed_url: Optional[str], episode_guid: Optional[str] = None,
    episode_url: Optional[str] = None, episode_title: Optional[str] = None,
    episode_published: Optional[str] = None, enclosure_url: Optional[str] = None,
    engine, transcribe_fn, **transcribe_kwargs,
) -> tuple[ExtractionResult, list[tuple[str, Path]]]:
    """Resolve a podcast enclosure, transcribe it, and wrap as ``audio_extraction``.

    Resolution precedence (plan §C):

    1. ``feed_url`` (+selector) → ``feed_parse`` (PRIMARY, RSS provenance).
    2. explicit ``enclosure_url`` → ``user_supplied`` (the only case that names a
       direct enclosure without feed identity; recorded honestly as such).

    A bare page URL is deliberately NOT accepted: yt-dlp's info.json cannot prove
    a concrete RSS enclosure/episode_guid, and the plan forbids silently minting
    weak podcast provenance — so such a job returns a structured
    ``feed_identity_unavailable`` error instead.

    Provenance lands on ``ExtractionResult.meta`` — its only byte-safe home —
    never on ``Transcript.meta``.
    """
    from .podcast import PodcastResolution, PodcastResolutionError, resolve_podcast

    if feed_url:
        resolution = resolve_podcast(feed_url, episode_guid=episode_guid,
                                     episode_url=episode_url, episode_title=episode_title,
                                     episode_published=episode_published)
    elif enclosure_url:
        # User-supplied enclosure: no provable feed identity, recorded explicitly.
        resolution = PodcastResolution(enclosure_url=enclosure_url,
                                       resolution_source="user_supplied")
    else:
        raise PodcastResolutionError(
            "feed_identity_unavailable",
            "audio_extraction needs a feed_url (+selector) or an explicit "
            "enclosure_url; a bare page URL cannot prove podcast provenance",
        )

    # Own the enclosure download (plan §C) so we learn the COMPLETE downloaded
    # size + redirect chain + Content-Length and can run the authoritative length
    # check. We do NOT fall back to yt-dlp on failure: yt-dlp is an opaque fetcher
    # that bypasses our SSRF host-block (a public enclosure could redirect to a
    # private IP) and our size cap — so a failed/blocked download fails the job.
    import shutil
    import tempfile

    from .podcast import (PodcastResolutionError as _PRErr, download_enclosure,
                          length_is_authoritative)

    feed_len = resolution.enclosure_length
    work = Path(tempfile.mkdtemp(prefix="enclosure-"))
    try:
        dl = download_enclosure(resolution.enclosure_url, work)
        if not dl.ok or not dl.path:
            raise _PRErr(
                "feed_identity_unavailable",
                f"could not download enclosure {resolution.enclosure_url!r}",
            )
        transcript: Transcript = transcribe_fn(str(dl.path), engine=engine,
                                               **transcribe_kwargs)

        downloaded_size = dl.downloaded_size
        content_length = dl.content_length
        # Authoritative only for a complete, non-ranged, non-redirected download —
        # exactly the case where a size mismatch is trustworthy (plan §C).
        authoritative = length_is_authoritative(
            redirected=bool(dl.redirect_chain), ranged=bool(dl.ranged),
            fully_downloaded=True,
        )
        length_matches = (downloaded_size == feed_len) if (downloaded_size is not None
                                                          and feed_len is not None) else None
        if authoritative and length_matches is False:
            # Fatal ONLY when the length is authoritative; stale/CDN-transformed
            # lengths stay observations (handled by length_matches above).
            raise _PRErr(
                "length_mismatch",
                f"downloaded {downloaded_size} bytes != feed <enclosure length> "
                f"{feed_len}",
            )
    finally:
        shutil.rmtree(work, ignore_errors=True)

    meta = dict(transcript.meta)  # lift ASR provenance onto the envelope
    # The legacy singular `selected_format` belongs only on Transcript.meta
    # (plan §B) — drop it from the envelope; podcast provenance is recorded below.
    meta.pop("selected_format", None)
    # `source` here is whatever was handed to transcribe() — for the direct-download
    # path that's a transient local temp file, which would leak server fs details
    # and make the envelope host/run-specific. The authoritative URLs
    # (enclosure_url / final_download_url) are recorded explicitly below.
    meta.pop("source", None)
    meta.update({
        "feed_url": resolution.feed_url,
        "episode_guid": resolution.episode_guid,
        "enclosure_url": resolution.enclosure_url,  # the feed's <enclosure>
        "final_download_url": (dl.final_url if dl else None),  # post-redirect
        "redirect_chain": (dl.redirect_chain if dl else []),
        "enclosure_length": feed_len,  # from the feed (observation)
        "content_length": content_length,  # response header (observation)
        "downloaded_size": downloaded_size,  # bytes actually fetched (observation)
        "length_authoritative": authoritative,
        "length_matches": length_matches,  # fatal only when authoritative (above)
        "enclosure_type": resolution.enclosure_type,
        "published": resolution.published,
        "resolution_source": resolution.resolution_source,
    })
    _tag_provenance(
        meta,
        recipe=["model", "device", "compute_type", "whisperx_version", "resolution_source"],
        observation=["feed_url", "episode_guid", "enclosure_url", "final_download_url",
                     "redirect_chain", "enclosure_length", "content_length",
                     "downloaded_size", "length_authoritative", "length_matches",
                     "enclosure_type", "published", "segments", "duration_s"],
    )
    result = ExtractionResult(
        kind="audio_extraction",
        text=render_audio_text(transcript),
        language=transcript.language,
        segments=transcript.segments,
        assets=[],
        meta=meta,
    )
    return result, []


# ---------------------------------------------------------------------------
# video — ASR (from the same bestaudio stream) + extracted frames + frame OCR
# ---------------------------------------------------------------------------


def extract_video(
    *, transcript: Transcript, video_path: Optional[Path], asset_dir: Path,
    with_frames: bool = True, cadence_s: Optional[float] = None, ocr_engine=None,
    selected_audio_format: Optional[str] = None,
    selected_video_format: Optional[str] = None,
) -> tuple[ExtractionResult, list[tuple[str, Path]]]:
    """Build a ``video`` envelope from an already-produced ASR ``transcript``,
    optionally with frames extracted from ``video_path``.

    ``with_frames`` is the orthogonal ``--frames`` switch (plan §Interface): when
    off, the envelope carries the ASR segments/text with ``frames: []`` and no
    frame assets. The ASR audio is taken from the same ``bestaudio`` stream the
    legacy path uses (caller's responsibility), so a video job and an audio job
    for one URL yield the SAME transcript.
    """
    from .frames import FRAME_POLICY, extract_frames, round_timecode
    from .ocr import run_ocr

    cadence = cadence_s if cadence_s is not None else FRAME_POLICY["cadence_s"]
    frame_assets = (
        extract_frames(video_path, asset_dir / "_frames", cadence_s=cadence)
        if (with_frames and video_path is not None) else []
    )

    frames: list[Frame] = []
    assets: list[AssetRef] = []
    asset_files: list[tuple[str, Path]] = []
    for fa in frame_assets:
        key = f"assets/frame-{fa.frame_id:06d}.jpg"
        try:
            ocr = run_ocr(fa.path, engine=ocr_engine)
            ocr_text, conf, blocks = ocr.ocr_text, ocr.confidence, ocr.blocks
        except Exception as exc:  # noqa: BLE001 — frames are useful without OCR
            log.warning("Frame OCR failed for %s: %s", fa.path.name, exc)
            ocr_text, conf, blocks = "", None, None
        frames.append(Frame(frame_id=fa.frame_id, timecode=round_timecode(fa.timecode),
                            image_ref=key, ocr_text=ocr_text, confidence=conf, blocks=blocks))
        assets.append(_asset_ref(key, fa.path))
        asset_files.append((key, fa.path))

    meta = dict(transcript.meta)
    # The legacy singular `selected_format` stays only on Transcript.meta (plan
    # §B); the video envelope exposes the explicit split fields instead.
    meta.pop("selected_format", None)
    # Drop the transient `source` (a server temp path for uploaded video) so the
    # envelope never leaks a server fs path — mirrors the audio_extraction path.
    meta.pop("source", None)
    if with_frames:
        from . import _ffmpeg_version
        policy = dict(FRAME_POLICY)
        policy["cadence_s"] = cadence
        meta.update({
            "frame_policy": policy,
            # §B provenance: present for local-file jobs too (URL jobs already
            # carry it via the download recipe; setdefault avoids overwriting).
            "selected_video_format": selected_video_format,
        })
        meta.setdefault("ffmpeg_version", _ffmpeg_version())
    meta.update({
        "frame_count": len(frames),
        "selected_audio_format": selected_audio_format,
    })
    # Only name recipe keys that are actually present (frame_policy /
    # selected_video_format exist only when frames were extracted).
    recipe = ["model", "selected_audio_format"]
    if with_frames:
        recipe += ["frame_policy", "selected_video_format"]
    _tag_provenance(
        meta, recipe=recipe,
        observation=["frames", "segments", "frame_count", "duration_s"],
    )
    result = ExtractionResult(
        kind="video",
        text=render_audio_text(transcript),
        language=transcript.language,
        segments=transcript.segments,
        frames=frames,
        assets=assets,
        meta=meta,
    )
    return result, asset_files
