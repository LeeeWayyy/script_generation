"""Core data structures returned by the transcription pipeline.

These are plain dataclasses with no heavy dependencies, so they can be imported
and inspected without loading torch/whisperx.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
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


def is_windows_reserved_basename(name: str) -> bool:
    """True when a basename resolves to a Windows device rather than a file."""
    # Windows ignores suffixes after a device stem and normalizes trailing dots /
    # spaces. A colon can introduce a device stream (``CON:x``).
    stem = name.rstrip(" .").split(".", 1)[0].rstrip(" ").split(":", 1)[0].upper()
    if stem in {"CON", "PRN", "AUX", "NUL", "CLOCK$", "CONIN$", "CONOUT$"}:
        return True
    return (len(stem) > 3 and stem[:3] in {"COM", "LPT"}
            and stem[3:].isdigit())


def word_offsets(text: str, words) -> Optional[list[tuple[int, int]]]:
    """Exact ``(start, end)`` character span of each word in ``text``.

    Gaps between words may hold only whitespace/punctuation (aligners drop
    characters outside their vocabulary); any unmatched letter or digit means
    the words don't account for the text, so return None rather than guess.
    """
    spans, cursor = [], 0
    for word in words:
        position = text.find(word.word, cursor) if word.word.strip() else -1
        if position < 0 or re.search(r"\w", text[cursor:position]):
            return None
        cursor = position + len(word.word)
        spans.append((position, cursor))
    return None if re.search(r"\w", text[cursor:]) else spans


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
    words: list[Word] = field(default_factory=list)
    # True when the segment overlaps detected music (sung lyrics or a song
    # playing under speech) — keyword-only so the legacy fifth positional
    # argument remains ``words``.
    music: bool = field(default=False, kw_only=True)


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

    def to_dict(self, *, offsets: bool = False) -> dict:
        """Frozen legacy JSON shape used by ``transcript[-remote] -f json``.

        Keep this explicit: new dataclass fields belong in versioned extraction
        envelopes, not in the byte-stable legacy output. ``offsets`` (opt-in
        reading units only) adds each word's ``char_start``/``char_end`` in its
        segment's ``text``, null when the words don't map onto the text.
        """
        def segment_dict(segment):
            spans = word_offsets(segment.text, segment.words) if offsets else None
            return {
                "text": segment.text,
                "start": segment.start,
                "end": segment.end,
                "speaker": segment.speaker,
                "words": [
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "score": word.score,
                        "speaker": word.speaker,
                        **({"char_start": spans[i][0] if spans else None,
                            "char_end": spans[i][1] if spans else None} if offsets else {}),
                    }
                    for i, word in enumerate(segment.words)
                ],
            }

        return {
            "segments": [segment_dict(segment) for segment in self.segments],
            "language": self.language,
            "meta": self.meta,
        }
