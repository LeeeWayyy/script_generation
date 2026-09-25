"""Opt-in reading units; never rewrite speech or interpolate word timing."""
from collections import Counter
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
    result = replace(transcript, segments=segments, meta={
        **transcript.meta,
        "readable": {"version": 1, "punctuation_restored": False, "fallbacks": fallback},
    })
    return _join_sentence_continuations(result)


def _join_sentence_continuations(transcript: Transcript) -> Transcript:
    """Treat a brief mid-sentence label change as uncertain, not an infallible turn.

    ponytail: conservative English punctuation/case heuristic, not a speaker
    classifier. Broader languages/longer sentences need acoustic boundary evidence.
    """
    if (transcript.language or "").split("-")[0].lower() != "en":
        return transcript
    interjections = {
        "yes", "no", "yeah", "yep", "nope", "oh", "ok", "okay", "right", "sure",
        "wow", "thanks", "huh", "what", "why", "well", "hey", "sorry", "wait",
        "stop", "go", "really", "exactly", "hmm", "uh", "um",
    }
    start_counts = Counter(s.start for s in transcript.segments)
    sentence, boundaries = [], []
    for segment in transcript.segments:
        sentence.append(segment)
        if not re.search(r'[.!?。！？]["\u201d\u2019]*$', segment.text):
            continue
        group, sentence = sentence, []
        if len(group) < 2 or len(group) > 3 or not segment.text.endswith("."):
            continue
        words = [w for s in group for w in s.words]
        if (not 2 <= len(words) <= 8 or not group[0].text[:1].isupper()
                or any(not s.words or s.speaker is None for s in group)
                or any(not isinstance(t, (int, float)) or not math.isfinite(t)
                       for w in words for t in (w.start, w.end))
                or any(w.start < 0 or w.end < w.start for w in words)
                or any(start_counts[s.start] != 1 for s in group[1:])
                or words[-1].end - words[0].start > 1.25
                or any(not 0 <= b.start - a.end <= .12 + 1e-6
                       for a, b in zip(words, words[1:]))):
            continue
        # Preserve explicit interjections and punctuation-delimited exchanges,
        # even when both are shorter than the unstable fragment being repaired.
        if any(s.words[0].word.lower().strip(".,!?;:\"'") in interjections for s in group):
            continue
        joins = []
        for left, right in zip(group, group[1:]):
            if (left.speaker == right.speaker or not right.text[:1].islower()
                    or re.search(r'[,;:\-—]$', left.text)
                    or min(left.words[-1].end - left.words[0].start,
                           right.words[-1].end - right.words[0].start) > .30 + 1e-6):
                break
            joins.append(right.start)
        else:
            boundaries.extend(joins)
    return _join_boundaries(transcript, boundaries, basis="short_sentence_continuity_heuristic")


def join_reviewed_boundaries(transcript: Transcript, boundaries: list[float]) -> Transcript:
    """Join only explicitly reviewed row boundaries, identified by the right row's start.

    Apply to an already-readable result. This is a human correction, not speaker
    inference: original word assignments survive, and conflicting row labels become null.
    """
    return _join_boundaries(transcript, boundaries, basis="user_confirmed_sentence_continuity")


def _join_boundaries(transcript: Transcript, boundaries: list[float], *, basis: str) -> Transcript:
    if not boundaries:
        return transcript
    if (any(not isinstance(t, (int, float)) or isinstance(t, bool)
            or not math.isfinite(t) or t < 0 for t in boundaries)
            or len(set(boundaries)) != len(boundaries)):
        raise ValueError("Reviewed boundaries must be unique finite nonnegative times")
    targets = set()
    for boundary in boundaries:
        matches = [i for i, s in enumerate(transcript.segments) if s.start == boundary]
        if len(matches) != 1 or matches[0] == 0:
            raise ValueError(f"Boundary {boundary} must identify exactly one non-first row")
        targets.add(matches[0])
    segments, indices, corrections = [], {}, []
    for i, source in enumerate(transcript.segments):
        if i not in targets:
            segments.append(source)
        else:
            left = segments[-1]
            words = left.words + source.words
            # No timing guesses or ASR-score-as-speaker-confidence. A reviewed
            # join still needs intact, monotonic word intervals for precise seeking.
            valid = all(
                isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t)
                for w in words for t in (w.start, w.end)
            )
            if (not left.words or not source.words or not valid
                    or any(w.start < 0 or w.start > w.end for w in words)
                    or any(b.start < a.end for a, b in zip(words, words[1:]))
                    or left.start != words[0].start or left.end != left.words[-1].end
                    or source.start != source.words[0].start or source.end != words[-1].end):
                raise ValueError(f"Boundary {source.start} lacks intact ordered word timing")
            labels = {w.speaker for w in words}
            label = next(iter(labels)) if len(labels) == 1 else None
            segments[-1] = replace(
                left, text=left.text.rstrip() + " " + source.text.lstrip(), end=source.end,
                words=words, speaker=label, music=left.music or source.music,
            )
            corrections.append({
                "segment_index": len(segments) - 1, "boundary_start": source.start,
                "input_segment_indices": [i - 1, i],
                "basis": basis,
                "speaker_assignments_changed": False,
            })
        indices[i] = len(segments) - 1
    readable = dict(transcript.meta.get("readable", {}))
    fallbacks = [{**f, "segment_index": indices[f["segment_index"]]}
                 for f in readable.get("fallbacks", [])]
    reviewed_indices = {c["segment_index"] for c in corrections}
    for index in sorted(reviewed_indices):
        segment = segments[index]
        if segment.speaker is None:
            fallbacks.append({
                "segment_index": index, "source_start": segment.start, "source_end": segment.end,
                "timing": "word", "speaker": "unavailable",
                "reason": basis + "_with_uncertain_speaker_assignment",
            })
    readable["fallbacks"] = fallbacks
    for key in ("reviewed_joins", "sentence_continuity_joins"):
        if key in readable:
            readable[key] = [{**c, "segment_index": indices[c["segment_index"]]}
                             for c in readable[key]]
    key = ("reviewed_joins" if basis == "user_confirmed_sentence_continuity"
           else "sentence_continuity_joins")
    readable[key] = readable.get(key, []) + corrections
    return replace(transcript, segments=segments, meta={**transcript.meta, "readable": readable})


def main():
    """Repair an exported readable JSON file without modifying the server or library."""
    import argparse
    import json
    from pathlib import Path
    from .types import Segment, Word

    parser = argparse.ArgumentParser(description=join_reviewed_boundaries.__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--join-at", type=float, nargs="+", required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    transcript = Transcript(
        segments=[Segment(**{**s, "words": [Word(**w) for w in s.get("words", [])]})
                  for s in data["segments"]], language=data.get("language"), meta=data.get("meta", {}),
    )
    try:
        result = join_reviewed_boundaries(transcript, args.join_at)
        output = json.dumps(result.to_dict(), indent=2, ensure_ascii=False, allow_nan=False)
        # Never overwrite a saved result or an active library, even if paths alias.
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(output + "\n")
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
