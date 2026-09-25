"""Opt-in reading units; never rewrite speech or interpolate word timing."""
from dataclasses import asdict, replace
import math
import re

from .types import Transcript


def readable_transcript(transcript: Transcript) -> Transcript:
    segments, fallback = [], []
    for source_index, source in enumerate(transcript.segments):
        words = source.words
        # Use word offsets only when they account for all spoken text.
        positions, cursor = [], 0
        for word in words:
            position = source.text.find(word.word, cursor) if word.word else -1
            if position < 0 or source.text[cursor:position].strip():
                break
            positions.append(position)
            cursor = position + len(word.word)
        mapped = bool(words) and len(positions) == len(words) and not source.text[cursor:].strip()
        tokens = [w.word for w in words] if mapped else re.findall(r"\S+", source.text)
        if not mapped:
            positions = [m.start() for m in re.finditer(r"\S+", source.text)]
        if not tokens:
            segments.append(replace(source))
            continue

        def timed(index):
            if not mapped:
                return False
            w = words[index]
            return (isinstance(w.start, (int, float)) and isinstance(w.end, (int, float))
                    and math.isfinite(w.start) and math.isfinite(w.end)
                    and 0 <= w.start <= w.end)

        def speaker(index):
            return words[index].speaker if mapped else source.speaker

        # ponytail: punctuation/pause/length heuristics; use a language-aware
        # segmenter only if these conservative reading boundaries prove inadequate.
        starts = [0]
        for i in range(1, len(tokens)):
            first = starts[-1]
            pause = timed(i - 1) and timed(i) and words[i].start - words[i - 1].end >= 0.8
            duration = timed(first) and timed(i) and words[i].end - words[first].start > 8
            if (speaker(i) != speaker(i - 1) or pause or duration or i - first >= 20
                    or positions[i] - positions[first] >= 120
                    or re.search(r'[.!?。！？]["\u201d\u2019]*$', tokens[i - 1])):
                starts.append(i)
        stops = starts[1:] + [len(tokens)]
        for first, stop in zip(starts, stops):
            group_words = words[first:stop] if mapped else []
            reliable = (mapped and all(timed(i) for i in range(first, stop))
                        and all(words[i].start >= words[i - 1].end
                                for i in range(first + 1, stop)))
            split = len(starts) > 1
            start = words[first].start if reliable else (None if split else source.start)
            end = words[stop - 1].end if reliable else (None if split else source.end)
            label = speaker(first) if mapped else source.speaker
            segments.append(replace(
                source, text=source.text[positions[first] if first else 0:
                                         positions[stop] if stop < len(tokens) else len(source.text)].strip(),
                start=start, end=end, speaker=label, words=group_words,
            ))
            if not reliable or label is None:
                fallback.append({
                    **({"source_words": [asdict(w) for w in words]}
                       if words and not mapped and first == 0 else {}),
                    "segment_index": len(segments) - 1, "source_segment_index": source_index,
                    "source_start": source.start, "source_end": source.end,
                    "timing": "word" if reliable else ("unavailable" if split else "source_segment"),
                    "speaker": "word" if mapped and label else (
                        "source_segment" if label else "unavailable"),
                })
    return replace(transcript, segments=segments, meta={
        **transcript.meta,
        "readable": {"version": 1, "punctuation_restored": False, "fallbacks": fallback},
    })
