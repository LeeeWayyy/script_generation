"""WhisperX wrapper: transcribe -> align -> diarize -> structured Transcript.

whisperx (and its torch dependency) are imported lazily inside functions so that
importing this package — or running ``transcript --help`` — does not pull in the
multi-gigabyte ML stack.
"""

from __future__ import annotations

import logging
import math
import os
import re
from bisect import bisect_right
from typing import Optional

from .device import default_compute_type, detect_device
from .types import Segment, Transcript, Word

log = logging.getLogger("transcript.engine")

DEFAULT_MODEL = "large-v3"
DIARIZATION_MODEL_URL = "https://huggingface.co/pyannote/speaker-diarization-community-1"
LEGACY_DIARIZATION_MODEL_URL = "https://huggingface.co/pyannote/speaker-diarization-3.1"


class TranscriptionEngine:
    """Loads WhisperX models once and reuses them across calls.

    The most recent alignment model is reused without retaining one model per
    language for the lifetime of a multilingual server.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        batch_size: int = 16,
        beam_size: int = 5,
        hf_token: Optional[str] = None,
    ):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(beam_size, bool) or not isinstance(beam_size, int) or beam_size <= 0:
            raise ValueError("beam_size must be a positive integer")
        self.device = detect_device(device)
        self.compute_type = compute_type or default_compute_type(self.device)
        self.model_name = model
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

        self._asr = None
        self._align_cache: Optional[tuple[str, tuple]] = None
        self._diarizer = None

        log.info(
            "Engine: model=%s device=%s compute_type=%s", self.model_name, self.device, self.compute_type
        )

    # --- model loaders (lazy) -------------------------------------------------

    def require_diarization_token(self) -> None:
        if not self.hf_token:
            raise RuntimeError(
                "Speaker diarization needs a Hugging Face token. Set HF_TOKEN (or pass "
                "hf_token=...), and accept the model conditions required by your "
                f"WhisperX version: {DIARIZATION_MODEL_URL} (3.8+) or "
                f"{LEGACY_DIARIZATION_MODEL_URL} (older versions)"
            )

    def _load_asr(self):
        if self._asr is None:
            import whisperx
            from .speech import SpeechVad

            try:
                self._asr = whisperx.load_model(
                    self.model_name,
                    self.device,
                    compute_type=self.compute_type,
                    asr_options={"beam_size": self.beam_size},
                    vad_model=SpeechVad(),
                )
            except ValueError as exc:
                # Some CPU builds reject float16; retry with int8 transparently.
                if self.device == "cpu" and self.compute_type != "int8":
                    log.warning("compute_type %s unsupported on CPU; retrying with int8.", self.compute_type)
                    self.compute_type = "int8"
                    self._asr = whisperx.load_model(
                        self.model_name,
                        self.device,
                        compute_type="int8",
                        asr_options={"beam_size": self.beam_size},
                        vad_model=SpeechVad(),
                    )
                else:
                    raise exc
        return self._asr

    def warm(self) -> None:
        """Load ASR/VAD weights now instead of delaying work until the first job."""
        self._load_asr()

    def _load_align(self, language: str):
        if self._align_cache is not None and self._align_cache[0] == language:
            return self._align_cache[1]
        import whisperx

        model = whisperx.load_align_model(language_code=language, device=self.device)
        self._align_cache = (language, model)
        return model

    def _load_diarizer(self):
        if self._diarizer is None:
            self.require_diarization_token()
            # DiarizationPipeline moved between whisperx versions; try both locations.
            try:
                from whisperx.diarize import DiarizationPipeline
            except ImportError:
                from whisperx import DiarizationPipeline  # older layout

            # The auth kwarg was renamed use_auth_token -> token in newer whisperx.
            try:
                self._diarizer = DiarizationPipeline(token=self.hf_token, device=self.device)
            except TypeError:
                self._diarizer = DiarizationPipeline(
                    use_auth_token=self.hf_token, device=self.device
                )
        return self._diarizer

    # --- main pipeline --------------------------------------------------------

    def run(
        self,
        audio_path: str,
        *,
        diarize: bool = True,
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        align: bool = True,
    ) -> Transcript:
        """Transcribe a 16 kHz mono WAV at ``audio_path`` into a Transcript."""
        if diarize:
            self.require_diarization_token()
        import whisperx

        audio = whisperx.load_audio(audio_path)

        log.info("Transcribing ...")
        asr = self._load_asr()
        result = asr.transcribe(audio, batch_size=self.batch_size, language=language)
        if not any(s.get("text", "").strip() for s in result.get("segments", [])):
            raise RuntimeError(
                "No reliable speech activity was detected; refusing to publish "
                "unverified captions. Check that the source contains audible speech."
            )
        detected_language = result.get("language", language)

        return self._align_and_diarize(
            audio, result, language=detected_language, diarize=diarize,
            min_speakers=min_speakers, max_speakers=max_speakers, align=align,
        )

    def run_captions(
        self,
        audio_path: str | None,
        captions: list[Segment],
        *,
        language: str,
        diarize: bool,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        align: bool = True,
    ) -> Transcript:
        """Use human caption text, running only alignment/diarization when requested."""
        if diarize:
            self.require_diarization_token()
        result = {
            "segments": [
                {"text": segment.text, "start": segment.start, "end": segment.end}
                for segment in captions
            ],
            "language": language,
        }
        if audio_path is None:
            return _stamp(_to_transcript(result, language=language), align=align,
                          align_ok=None, diarize=False)

        import whisperx

        audio = whisperx.load_audio(audio_path)
        return self._align_and_diarize(
            audio, result, language=language, diarize=diarize,
            min_speakers=min_speakers, max_speakers=max_speakers, align=align,
        )

    def _align_and_diarize(
        self, audio, result: dict, *, language: Optional[str], diarize: bool,
        min_speakers: Optional[int], max_speakers: Optional[int], align: bool,
    ) -> Transcript:
        import whisperx

        align_ok: Optional[bool] = None
        alignment_adjustments = []
        if align and language:
            align_ok = False
            try:
                log.info("Aligning words (language=%s) ...", language)
                model_a, metadata = self._load_align(language)
                result = whisperx.align(
                    result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False
                )
                alignment_adjustments = _deduplicate_alignment_boundaries(result)
                align_ok = True
            except Exception as exc:
                log.warning("Word alignment failed (%s); continuing without word timestamps.", exc)

        timing_adjustments = []
        if diarize:
            log.info("Diarizing (identifying speakers) ...")
            diarizer = self._load_diarizer()
            diarize_segments = diarizer(audio, min_speakers=min_speakers, max_speakers=max_speakers)
            if len(diarize_segments["start"]) == 0:
                raise RuntimeError(
                    "No reliable speech activity was detected; refusing to publish "
                    "unverified captions. Check that the source contains audible speech."
                )
            timing_adjustments = _trim_sentence_tails(result, diarize_segments)
            result = whisperx.assign_word_speakers(diarize_segments, result)

        transcript = _stamp(_to_transcript(result, language=language), align=align,
                            align_ok=align_ok, diarize=diarize)
        if timing_adjustments:
            transcript.meta["timing_adjustments"] = timing_adjustments
        if alignment_adjustments:
            transcript.meta["alignment_adjustments"] = alignment_adjustments
        return transcript


def _deduplicate_alignment_boundaries(result: dict) -> list[dict]:
    """Remove an aligner's inclusive-end character repeated in the next sentence.

    Require exact text coverage and identical word evidence in the adjacent row;
    never discard an unmatched word merely to make the readable mapper succeed.
    """
    adjustments = []
    segments = result.get("segments", [])
    for index, (left, right) in enumerate(zip(segments, segments[1:])):
        words, following = left.get("words", []), right.get("words", [])
        if len(words) < 2 or not following or words[-1] != following[0]:
            continue
        duplicate = words[-1]
        text = "".join(left.get("text", "").split())
        covered = "".join("".join(w.get("word", "") for w in words[:-1]).split())
        end, next_start = words[-2].get("end"), duplicate.get("start")
        if (not text or covered != text or not duplicate.get("word", "").strip()
                or not right.get("text", "").lstrip().startswith(duplicate["word"])
                or any(not isinstance(t, (int, float)) or isinstance(t, bool)
                       or not math.isfinite(t) for t in (end, next_start))
                or end > next_start):
            continue
        adjustments.append({"source_segment_index": index,
                            "reason": "duplicated_alignment_boundary_character",
                            "original_end": left.get("end"), "end": end,
                            "duplicate_word": dict(duplicate)})
        left["words"] = words[:-1]
        left["end"] = end
    if adjustments and "word_segments" in result:
        result["word_segments"] = [w for s in segments for w in s.get("words", [])]
    return adjustments


def _trim_sentence_tails(result: dict, diarization) -> list[dict]:
    """Stop a stretched final word from including the next speech island.

    Uses gaps in the union of all speakers, never ASR score as speaker confidence.
    Original alignment is retained in provenance; these remain model estimates.
    """
    def valid(t):
        return isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t)

    islands = []
    for start, end in sorted(zip(diarization["start"], diarization["end"])):
        if not valid(start) or not valid(end) or not 0 <= start < end:
            continue
        if islands and start <= islands[-1][1]:
            islands[-1][1] = max(islands[-1][1], end)
        else:
            islands.append([start, end])
    starts = [span[0] for span in islands]
    adjustments = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            start, end = word.get("start"), word.get("end")
            if (not valid(start) or not valid(end) or not .5 <= end - start <= 1.
                    or not re.search(r'[.!?。！？]["\u201d\u2019]*$', word.get("word", ""))):
                continue
            index = bisect_right(starts, start) - 1
            if index < 0 or index + 1 >= len(islands):
                continue
            stop = islands[index][1]
            next_start = starts[index + 1]
            # ponytail: >=200ms all-speaker gap and 40–300ms retained speech;
            # finer boundaries need calibrated frame/phoneme evidence.
            if not .04 <= stop - start <= .3 or next_start - stop < .2 or end <= next_start:
                continue
            word["end"] = stop
            if segment.get("end") == end:
                segment["end"] = stop
            adjustments.append({
                "word": word["word"], "start": start, "original_end": end, "end": stop,
                "reason": "sentence_tail_crossed_diarization_speech_gap",
                "timing_precision": "model_estimate",
            })
    return adjustments


def _pkg_version(name: str) -> Optional[str]:
    """Installed package version (for provenance); None if unknown."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _stamp(t: Transcript, *, align: bool, align_ok: Optional[bool], diarize: bool) -> Transcript:
    """Attach the engine provenance shared by ASR and human-caption paths."""
    ends = [s.end for s in t.segments if s.end is not None]
    t.meta.update(
        {
            "align_requested": align,
            "align_succeeded": align_ok,
            "diarize_requested": diarize,
            "diarize_succeeded": (any(s.speaker for s in t.segments) if diarize else None),
            "duration_s": (max(ends) if ends else None),
            "whisperx_version": _pkg_version("whisperx"),
            "pyannote_version": _pkg_version("pyannote.audio"),
        }
    )
    return t


def _split_speaker_turns(raw: dict) -> list[dict]:
    """Split an aligned segment at word-level speaker changes without rewriting text."""
    text = raw.get("text", "")
    words = raw.get("words", [])
    if not text or not words:
        return [raw]

    speakers = [word.get("speaker") for word in words]
    if len({speaker for speaker in speakers if speaker}) < 2:
        return [raw]

    positions = []
    cursor = 0
    for word in words:
        token = word.get("word", "")
        position = text.find(token, cursor)
        if not token or position < 0:
            return [raw]
        positions.append(position)
        cursor = position + len(token)

    groups: list[tuple[int, int, Optional[str]]] = []
    start = 0
    last_speaker = speakers[0]
    for index in range(1, len(words)):
        if speakers[index] is not None and last_speaker is not None and speakers[index] != last_speaker:
            groups.append((start, index, last_speaker))
            start = index
        if speakers[index] is not None:
            last_speaker = speakers[index]
    groups.append((start, len(words), last_speaker))

    split = []
    for group_index, (word_start, word_end, speaker) in enumerate(groups):
        char_start = 0 if group_index == 0 else positions[word_start]
        char_end = len(text) if group_index == len(groups) - 1 else positions[word_end]
        group_words = words[word_start:word_end]
        piece = dict(raw)
        piece.update({
            "text": text[char_start:char_end].strip(),
            "start": next((word.get("start") for word in group_words
                           if word.get("start") is not None), raw.get("start")),
            "end": next((word.get("end") for word in reversed(group_words)
                         if word.get("end") is not None), raw.get("end")),
            "speaker": speaker if all(w.get("speaker") == speaker for w in group_words) else None,
            "words": group_words,
        })
        split.append(piece)
    return split


def _to_transcript(result: dict, *, language: Optional[str]) -> Transcript:
    """Convert a raw whisperx result dict into our Transcript dataclass."""
    segments: list[Segment] = []
    raw_segments = [
        split for raw in result.get("segments", []) for split in _split_speaker_turns(raw)
    ]
    for raw in raw_segments:
        words = [
            Word(
                word=w.get("word", ""),
                start=w.get("start"),
                end=w.get("end"),
                score=w.get("score"),
                speaker=w.get("speaker"),
            )
            for w in raw.get("words", [])
        ]
        segments.append(
            Segment(
                text=raw.get("text", "").strip(),
                start=raw.get("start"),
                end=raw.get("end"),
                speaker=raw.get("speaker"),
                words=words,
            )
        )
    return Transcript(segments=segments, language=language)
