"""transcript — generate transcripts (with speaker labels) from any video/audio source.

Public API
----------
    from transcript import transcribe
    t = transcribe("meeting.mp4", diarize=True)
    print(t.text)
    print(t.to_dict())

    # or from a URL
    t = transcribe("https://www.youtube.com/watch?v=...", diarize=True)
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .audio import extract_audio
from .engine import DEFAULT_MODEL, TranscriptionEngine
from .ingest import (
    download_manual_caption,
    is_url,
    is_youtube_url,
    parse_json3_caption,
    resolve_source,
)
from .types import Segment, Transcript, Word

__all__ = [
    "transcribe",
    "TranscriptionEngine",
    "Transcript",
    "Segment",
    "Word",
    "DEFAULT_MODEL",
]

__version__ = "0.1.0"

log = logging.getLogger("transcript")


def _validate_transcribe_options(
    *, diarize: bool, device: Optional[str], min_speakers: Optional[int],
    max_speakers: Optional[int], batch_size: int,
) -> None:
    if device not in (None, "cpu", "cuda", "mps"):
        raise ValueError("device must be one of: cpu, cuda, mps")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    for name, value in (("min_speakers", min_speakers), ("max_speakers", max_speakers)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"{name} must be at least 1")
    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise ValueError("min_speakers cannot exceed max_speakers")
    if not diarize and (min_speakers is not None or max_speakers is not None):
        raise ValueError("speaker-count hints cannot be used when diarize is false")


def _ffmpeg_version() -> Optional[str]:
    """ffmpeg version string for provenance (None if unavailable)."""
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split()
        return out[2] if len(out) >= 3 and out[0] == "ffmpeg" else None
    except (OSError, IndexError):
        return None


def transcribe(
    source: str,
    *,
    model: str = DEFAULT_MODEL,
    diarize: bool = True,
    language: Optional[str] = None,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    hf_token: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    batch_size: int = 16,
    align: bool = True,
    detect_music: bool = False,
    work_dir: Optional[str] = None,
    keep_audio: bool = False,
    engine: Optional[TranscriptionEngine] = None,
) -> Transcript:
    """Transcribe a local media file or URL into a :class:`Transcript`.

    Parameters
    ----------
    source        Local path or http(s) URL (anything yt-dlp supports).
    model         Whisper model name (e.g. "large-v3", "medium", "small").
    diarize       Attach speaker labels (needs a Hugging Face token).
    language      Force a language code (e.g. "en"); None auto-detects.
    device        "cuda" or "cpu"; None auto-detects (CUDA if available, else CPU).
    compute_type  CTranslate2 compute type; None picks float16 (CUDA) / int8 (CPU).
    hf_token      Hugging Face token for diarization; falls back to $HF_TOKEN.
    min/max_speakers  Optional hints to the diarizer.
    detect_music  Opt in to music-overlap tagging (requires inaSpeechSegmenter).
    engine        Reuse a pre-built TranscriptionEngine (avoids reloading models).
    """
    _validate_transcribe_options(
        diarize=diarize, device=device, min_speakers=min_speakers,
        max_speakers=max_speakers, batch_size=batch_size,
    )
    eng = engine or TranscriptionEngine(
        model=model,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        hf_token=hf_token,
    )
    if diarize and type(eng) is TranscriptionEngine:
        eng.require_diarization_token()

    own_work_dir = work_dir is None
    work_path = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="transcript-"))
    needs_audio = align or diarize or detect_music

    try:
        caption_download = (
            download_manual_caption(
                source, work_path, language=language, with_audio=needs_audio,
            )
            if is_youtube_url(source) else None
        )
        caption_path = caption_download[0] if caption_download else None
        caption_language = caption_download[1] if caption_download else None
        media = caption_download[2] if caption_download else None
        caption_info = caption_download[3] if caption_download else None

        captions = None
        if caption_path is not None:
            try:
                captions = parse_json3_caption(caption_path)
            except (OSError, TypeError, ValueError) as exc:
                log.warning("Could not parse manual captions (%s); falling back to ASR.", exc)
            if not captions:
                log.warning("Manual caption track was empty; falling back to ASR.")

        if captions:
            audio = extract_audio(media, work_path) if needs_audio else None
            alignment_language = caption_language.split("-", 1)[0]
            result = eng.run_captions(
                str(audio) if audio else None,
                captions,
                diarize=diarize,
                language=alignment_language,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                align=align,
            )
            result.meta.update({
                "transcript_source": "youtube_manual_captions",
                "caption_language": caption_language,
            })
        else:
            media = media or resolve_source(source, work_path)
            audio = extract_audio(media, work_path)
            result = eng.run(
                str(audio),
                diarize=diarize,
                language=language,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                align=align,
            )
        if detect_music:
            from .music import MIN_OVERLAP, detect_and_tag, detector_version
            flagged = detect_and_tag(result, str(audio))
            result.meta.update({
                "music_detection_requested": True,
                "music_detection_succeeded": flagged is not None,
                "music_detector_version": detector_version(),
                "music_overlap_threshold": MIN_OVERLAP,
                "music_segments_flagged": flagged,
            })
        result.meta.update(
            {
                "source": source,
                "model": eng.model_name,
                "device": eng.device,
                "compute_type": eng.compute_type,
                "diarized": diarize,
            }
        )
        # Download recipe (provenance): for a URL, merge yt-dlp's metadata
        # (written to `<id>.info.json` by --write-info-json) + tool versions.
        # No-op for local-file sources.
        if is_url(source):
            from importlib.metadata import PackageNotFoundError, version
            try:
                ytdlp_ver = version("yt-dlp")
            except PackageNotFoundError:
                ytdlp_ver = None
            rec = {
                "video_id": ((caption_info or {}).get("id")
                             or (Path(media).stem if media is not None else "")),
                "downloader": "yt-dlp",
                "yt_dlp_version": ytdlp_ver,
                "ffmpeg_version": _ffmpeg_version(),
            }
            info = {}
            info_path = (Path(media).with_name(f"{Path(media).stem}.info.json")
                         if media is not None else None)
            if info_path is not None and info_path.is_file():
                import json as _json
                try:
                    info = _json.loads(info_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    info = {}
            if not info and captions:
                info = caption_info or {}
            for key, meta_key in (
                ("id", "video_id"), ("webpage_url", "resolved_url"),
                ("format_id", "selected_format"), ("channel", "channel"),
                ("uploader", "uploader"), ("upload_date", "upload_date"),
            ):
                if info.get(key):
                    rec[meta_key] = info[key]
            result.meta.update(rec)
        return result
    finally:
        if own_work_dir and not keep_audio:
            shutil.rmtree(work_path, ignore_errors=True)
