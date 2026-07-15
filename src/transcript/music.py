"""Music detection: flag transcript segments that overlap sung/instrumental music.

Whisper transcribes lyrics as if they were dialogue and hallucinates during
instrumentals, so downstream consumers need to know which segments are
untrustworthy. inaSpeechSegmenter (lazy-imported, optional) labels the audio
timeline speech/music/noise; we intersect those music ranges with the
transcript segments.

When singing and speech overlap, Whisper emits a single text stream following
the dominant vocal — the `music` flag on such segments means "a song was
playing here; text and speaker label may be unreliable", not "this is lyrics".
"""

from __future__ import annotations

import logging
from functools import lru_cache

from .types import Transcript

log = logging.getLogger("transcript.music")

# A segment is flagged when at least this fraction of it lies inside music.
MIN_OVERLAP = 0.5


@lru_cache(maxsize=1)
def _segmenter():
    from inaSpeechSegmenter import Segmenter

    return Segmenter(vad_engine="smn", detect_gender=False)


def detect_music(audio_path: str) -> list[tuple[float, float]]:
    """Return (start, end) ranges of ``audio_path`` classified as music."""
    return [
        (start, end)
        for label, start, end in _segmenter()(audio_path)
        if label == "music"
    ]


def tag_music(transcript: Transcript, ranges: list[tuple[float, float]]) -> int:
    """Set ``segment.music = True`` where a segment overlaps a music range.

    Returns the number of segments flagged.
    """
    flagged = 0
    for seg in transcript.segments:
        if seg.start is None or seg.end is None or seg.end <= seg.start:
            continue
        overlap = sum(
            max(0.0, min(seg.end, r_end) - max(seg.start, r_start))
            for r_start, r_end in ranges
        )
        if overlap / (seg.end - seg.start) >= MIN_OVERLAP:
            seg.music = True
            flagged += 1
    return flagged


def detect_and_tag(transcript: Transcript, audio_path: str) -> None:
    """Best-effort music tagging: never fails the transcription pipeline."""
    try:
        ranges = detect_music(audio_path)
    except ImportError:
        log.info("inaSpeechSegmenter not installed; skipping music tagging.")
        return
    except Exception as exc:
        log.warning("Music detection failed (%s); continuing without music tags.", exc)
        return
    flagged = tag_music(transcript, ranges)
    log.info("Music: %d range(s), %d segment(s) flagged.", len(ranges), flagged)
