"""Render a Transcript into txt / srt / vtt / json output formats."""

from __future__ import annotations

import json
from typing import Optional

from .types import Transcript

FORMATS = ("txt", "srt", "vtt", "json")


def render(transcript: Transcript, fmt: str) -> str:
    fmt = fmt.lower()
    if fmt == "txt":
        return to_txt(transcript)
    if fmt == "srt":
        return to_srt(transcript)
    if fmt == "vtt":
        return to_vtt(transcript)
    if fmt == "json":
        return to_json(transcript)
    raise ValueError(f"Unknown format '{fmt}'. Choose from: {', '.join(FORMATS)}")


def to_txt(transcript: Transcript) -> str:
    """Plain text, prefixed with speaker labels when diarization is present."""
    lines: list[str] = []
    for seg in transcript.segments:
        text = seg.text.strip()
        if not text:
            continue
        if seg.speaker:
            lines.append(f"{seg.speaker}: {text}")
        else:
            lines.append(text)
    return "\n".join(lines) + "\n"


def to_srt(transcript: Transcript) -> str:
    blocks: list[str] = []
    for i, seg in enumerate(transcript.segments, start=1):
        start = _ts(seg.start, srt=True)
        end = _ts(seg.end, srt=True)
        text = _label(seg)
        blocks.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def to_vtt(transcript: Transcript) -> str:
    blocks = ["WEBVTT\n"]
    for seg in transcript.segments:
        start = _ts(seg.start, srt=False)
        end = _ts(seg.end, srt=False)
        text = _label(seg)
        blocks.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def to_json(transcript: Transcript) -> str:
    return json.dumps(transcript.to_dict(), indent=2, ensure_ascii=False)


def _label(seg) -> str:
    text = seg.text.strip()
    return f"[{seg.speaker}] {text}" if seg.speaker else text


def _ts(seconds: Optional[float], *, srt: bool) -> str:
    """Format seconds as a subtitle timestamp. srt uses ',' for ms, vtt uses '.'."""
    if seconds is None:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    sep = "," if srt else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"
