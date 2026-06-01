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
import tempfile
from pathlib import Path
from typing import Optional

from .audio import extract_audio
from .engine import DEFAULT_MODEL, TranscriptionEngine
from .ingest import resolve_source
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
    engine        Reuse a pre-built TranscriptionEngine (avoids reloading models).
    """
    own_work_dir = work_dir is None
    work_path = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="transcript-"))

    try:
        media = resolve_source(source, work_path)
        audio = extract_audio(media, work_path)

        eng = engine or TranscriptionEngine(
            model=model,
            device=device,
            compute_type=compute_type,
            batch_size=batch_size,
            hf_token=hf_token,
        )

        result = eng.run(
            str(audio),
            diarize=diarize,
            language=language,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            align=align,
        )
        result.meta.update(
            {
                "source": source,
                "model": eng.model_name,
                "device": eng.device,
                "compute_type": eng.compute_type,
                "diarized": diarize,
            }
        )
        # Download recipe (provenance): for a URL, yt-dlp named the media file
        # `<id>.<ext>`, so the stem is the resolved platform id; also record the
        # downloader version. (No-op for local-file sources.)
        if source.startswith(("http://", "https://")):
            from importlib.metadata import PackageNotFoundError, version
            try:
                ytdlp_ver = version("yt-dlp")
            except PackageNotFoundError:
                ytdlp_ver = None
            result.meta.update({"video_id": Path(media).stem, "downloader": "yt-dlp", "yt_dlp_version": ytdlp_ver})
        return result
    finally:
        if own_work_dir and not keep_audio:
            shutil.rmtree(work_path, ignore_errors=True)
