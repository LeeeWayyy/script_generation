"""WhisperX wrapper: transcribe -> align -> diarize -> structured Transcript.

whisperx (and its torch dependency) are imported lazily inside functions so that
importing this package — or running ``transcript --help`` — does not pull in the
multi-gigabyte ML stack.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .device import default_compute_type, detect_device
from .types import Segment, Transcript, Word

log = logging.getLogger("transcript.engine")

DEFAULT_MODEL = "large-v3"


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
        hf_token: Optional[str] = None,
    ):
        self.device = detect_device(device)
        self.compute_type = compute_type or default_compute_type(self.device)
        self.model_name = model
        self.batch_size = batch_size
        self.hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

        self._asr = None
        self._align_cache: Optional[tuple[str, tuple]] = None
        self._diarizer = None

        log.info(
            "Engine: model=%s device=%s compute_type=%s", self.model_name, self.device, self.compute_type
        )

    # --- model loaders (lazy) -------------------------------------------------

    def _load_asr(self):
        if self._asr is None:
            import whisperx

            try:
                self._asr = whisperx.load_model(
                    self.model_name, self.device, compute_type=self.compute_type
                )
            except ValueError as exc:
                # Some CPU builds reject float16; retry with int8 transparently.
                if self.device == "cpu" and self.compute_type != "int8":
                    log.warning("compute_type %s unsupported on CPU; retrying with int8.", self.compute_type)
                    self.compute_type = "int8"
                    self._asr = whisperx.load_model(self.model_name, self.device, compute_type="int8")
                else:
                    raise exc
        return self._asr

    def _load_align(self, language: str):
        if self._align_cache is not None and self._align_cache[0] == language:
            return self._align_cache[1]
        import whisperx

        model = whisperx.load_align_model(language_code=language, device=self.device)
        self._align_cache = (language, model)
        return model

    def _load_diarizer(self):
        if self._diarizer is None:
            if not self.hf_token:
                raise RuntimeError(
                    "Speaker diarization needs a Hugging Face token. Set HF_TOKEN (or pass "
                    "hf_token=...), and accept the model license at "
                    "https://huggingface.co/pyannote/speaker-diarization-3.1"
                )
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
        import whisperx

        audio = whisperx.load_audio(audio_path)

        log.info("Transcribing ...")
        asr = self._load_asr()
        result = asr.transcribe(audio, batch_size=self.batch_size, language=language)
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
        if align and language:
            align_ok = False
            try:
                log.info("Aligning words (language=%s) ...", language)
                model_a, metadata = self._load_align(language)
                result = whisperx.align(
                    result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False
                )
                align_ok = True
            except Exception as exc:
                log.warning("Word alignment failed (%s); continuing without word timestamps.", exc)

        if diarize:
            log.info("Diarizing (identifying speakers) ...")
            diarizer = self._load_diarizer()
            diarize_segments = diarizer(audio, min_speakers=min_speakers, max_speakers=max_speakers)
            result = whisperx.assign_word_speakers(diarize_segments, result)

        return _stamp(_to_transcript(result, language=language), align=align,
                      align_ok=align_ok, diarize=diarize)


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

    speakers = [word.get("speaker") or raw.get("speaker") for word in words]
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
    for index in range(1, len(words)):
        if speakers[index] != speakers[index - 1]:
            groups.append((start, index, speakers[index - 1]))
            start = index
    groups.append((start, len(words), speakers[-1]))

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
            "speaker": speaker,
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
