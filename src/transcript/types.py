"""Core data structures returned by the transcription pipeline.

These are plain dataclasses with no heavy dependencies, so they can be imported
and inspected without loading torch/whisperx.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Optional


def nfc(s: str) -> str:
    """NFC-normalize a string (the extraction/OCR text contract)."""
    return unicodedata.normalize("NFC", s)


def has_windows_drive_prefix(p: str) -> bool:
    """True if ``p`` starts with a Windows drive prefix — absolute (``C:/x``) or
    drive-relative (``C:x``). Both escape a destination dir on a Windows host, so
    the path-safety guards reject them even though only the former is
    ``os.path``-absolute. A leading ``0:00.jpg`` is fine (first char isn't alpha)."""
    return len(p) >= 2 and p[0].isalpha() and p[1] == ":"


@dataclass
class Word:
    """A single word with its timing and (optionally) the speaker who said it."""

    word: str
    start: Optional[float] = None
    end: Optional[float] = None
    score: Optional[float] = None
    speaker: Optional[str] = None


@dataclass
class Segment:
    """A contiguous chunk of speech (typically one sentence/utterance)."""

    text: str
    start: Optional[float] = None
    end: Optional[float] = None
    speaker: Optional[str] = None
    # True when the segment overlaps detected music (sung lyrics or a song
    # playing under speech) — treat text/speaker as unreliable there.
    music: bool = False
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    """The full result of transcribing one source."""

    segments: list[Segment] = field(default_factory=list)
    language: Optional[str] = None
    # Free-form provenance/info: source path/url, model, device, duration, etc.
    meta: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        """The plain, un-timestamped transcript text."""
        return "\n".join(seg.text.strip() for seg in self.segments if seg.text.strip())

    @property
    def speakers(self) -> list[str]:
        """Sorted list of distinct speaker labels present in the transcript."""
        found = {seg.speaker for seg in self.segments if seg.speaker}
        return sorted(found)

    def to_dict(self) -> dict:
        return asdict(self)
