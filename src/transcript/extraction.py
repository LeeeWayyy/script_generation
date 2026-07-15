"""The extraction envelope — media → candidate text (+ assets) + a provenance recipe.

This is the *new* result shape, kept strictly separate from the legacy
:class:`~transcript.types.Transcript` so the byte-stable ASR path
(``transcript-remote -f json`` → ``render(Transcript)``) is never touched.

Hard boundaries enforced here:

* **Vault-agnostic.** We return candidate ``text`` + asset files (opaque
  job-relative keys + integrity) + a ``meta`` recipe. We know nothing about
  sidecars/citations/dedup/wiki. The *consumer* commits + hashes to confer
  canonicality.
* **One explicit serializer.** :func:`serialize` pins field order, the
  None-vs-omit-vs-``[]`` policy, and NFC. FastAPI/Pydantic defaults are NOT
  used, so the durable ``result.json`` in the bundle is byte-identical to the
  ``/extractions/{id}/result`` route.
* **NFC is an extraction-renderer-only rule.** It is applied identically across
  the three ``text`` renderers and is NEVER applied inside ``formats.py`` /
  ``Transcript`` / ``Segment`` / the legacy ``/jobs/{id}/result`` route
  (normalizing there would change byte-stable ASR output).

No heavy dependencies (torch/whisperx/paddleocr/…) are imported here; the only
module-top imports are the stdlib-only ``.formats`` and ``.types``. NOTE: the
thin-client boundary is upheld by ``extract_remote.py`` NOT importing this module
at all — it is not, itself, on the client import path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from .formats import to_txt
from .types import Segment, Transcript, nfc as _nfc

# The three extraction kinds. Plain ASR ``audio`` is deliberately NOT a kind:
# it stays on the legacy Transcript route. ``audio_extraction`` (podcast/RSS) is
# spelled out so "audio" never names two commands that hash different bytes.
KINDS = ("video", "image_note", "audio_extraction")


# ---------------------------------------------------------------------------
# Asset integrity
# ---------------------------------------------------------------------------


@dataclass
class AssetRef:
    """An opaque job-relative pointer to a bundled asset, carrying integrity.

    ``key`` is neither a URL nor a server path (e.g. ``assets/card-000.jpg``).
    Every asset type carries ``sha256``/``size`` so a truncated/corrupt bundle
    cannot still "match" the envelope — the consumer verifies after unzip.
    """

    key: str
    sha256: str
    size: int
    media_type: str


@dataclass
class Card:
    """One OCR'd image in an ``image_note`` extraction.

    ``image_sha256`` is over the ORIGINAL bytes (for an archive input, the
    extracted member's original bytes — not a re-encode, not the enclosing
    archive). ``source_filename`` is the bare basename; ``source_member`` is the
    sanitized original archive member path, retained only as a debugging
    observation (never used for asset paths or ordering).
    """

    index: int
    ocr_text: str
    image_ref: str  # the AssetRef.key for this card's image
    source_filename: str
    image_sha256: str
    width: Optional[int] = None
    height: Optional[int] = None
    confidence: Optional[float] = None
    source_member: Optional[str] = None
    blocks: Optional[list[dict]] = None


@dataclass
class Frame:
    """One extracted video frame.

    Integrity is single-sourced: ``image_ref`` resolves to its :class:`AssetRef`
    (which carries ``sha256``/``size``) — there is deliberately no
    ``frames[].image_sha256`` (two hashes invite a mismatch with no winner).
    """

    frame_id: int  # the ordinal N
    timecode: float  # seconds on the *video* stream clock (see plan §B)
    image_ref: str
    ocr_text: str = ""
    confidence: Optional[float] = None
    blocks: Optional[list[dict]] = None


@dataclass
class ExtractionResult:
    """The separate envelope returned by every extraction kind."""

    kind: str  # one of KINDS
    text: str  # candidate canonical-text (NFC, via the named renderers below)
    language: Optional[str] = None
    segments: Optional[list[Segment]] = None  # audio_extraction / video
    frames: Optional[list[Frame]] = None  # video
    cards: Optional[list[Card]] = None  # image_note
    assets: list[AssetRef] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-modality text renderers (NFC) — named, tested, pinned. The consumer
# hashes `text`, so these are contract.
# ---------------------------------------------------------------------------


def render_audio_text(transcript: Transcript) -> str:
    """``audio_extraction``/``video`` text = NFC(to_txt(transcript)).

    Speaker prefixes, blank segments skipped, trailing ``\\n`` — then NFC. This
    is the NEW extraction renderer, NOT the legacy ``transcript-remote -f txt``
    code path.
    """
    return _nfc(to_txt(transcript))


def render_image_note_text(cards: list[Card]) -> str:
    """Render ``image_note`` cards into candidate text.

    Pinned literal rules (golden-tested):

    * zero cards → ``""`` (no trailing newline);
    * each card is ``## card N\\n<ocr_text>`` (1-based; internal OCR line-breaks
      preserved, trailing ones trimmed). An empty-OCR card is therefore
      ``## card N\\n`` (header + the template newline, no body);
    * cards joined by ``\\n\\n``;
    * the result ends with exactly **one** trailing ``\\n`` (cf. ``to_txt``'s
      ``+ "\\n"``). The plan's per-card literal (``## card N\\n``) and its
      "single trailing newline" rule only conflict for a *trailing empty* card;
      we resolve it by normalizing the final newline once, after the join, which
      satisfies both;
    * the whole result is NFC-normalized.

    ``N`` is the 1-based card number (``index`` is 0-based — pinned explicitly).
    """
    if not cards:
        return ""
    # Each card is verbatim `## card N\n<ocr_text>` — internal OCR line breaks
    # (including blank lines) are preserved, NOT stripped per-card. Only the ONE
    # global trailing newline is normalized (also folding a trailing CRLF), so the
    # candidate text the consumer hashes faithfully reflects the OCR output.
    blocks = [f"## card {n}\n{card.ocr_text}" for n, card in enumerate(cards, start=1)]
    joined = "\n\n".join(blocks)
    return _nfc(joined.rstrip("\r\n") + "\n")


# ---------------------------------------------------------------------------
# The one explicit serializer — byte-identical between /result and result.json.
# Field order, None-vs-omit-vs-[] policy, and NFC are all pinned here. Do NOT
# rely on FastAPI/Pydantic defaults.
# ---------------------------------------------------------------------------


def _segment_to_dict(seg: Segment) -> dict:
    """Pinned extraction segment shape (new fields do not enter legacy JSON)."""
    return {
        "text": seg.text,
        "start": seg.start,
        "end": seg.end,
        "speaker": seg.speaker,
        "words": [
            {
                "word": w.word,
                "start": w.start,
                "end": w.end,
                "score": w.score,
                "speaker": w.speaker,
            }
            for w in seg.words
        ],
        "music": seg.music,
    }


def _card_to_dict(card: Card) -> dict:
    d = {
        "index": card.index,
        "ocr_text": card.ocr_text,
        "image_ref": card.image_ref,
        "source_filename": card.source_filename,
        "image_sha256": card.image_sha256,
        "width": card.width,
        "height": card.height,
        "confidence": card.confidence,
    }
    # Optional observations: omit when absent so the common shape stays lean.
    if card.source_member is not None:
        d["source_member"] = card.source_member
    if card.blocks is not None:
        d["blocks"] = card.blocks
    return d


def _frame_to_dict(frame: Frame) -> dict:
    d = {
        "frame_id": frame.frame_id,
        "timecode": frame.timecode,
        "image_ref": frame.image_ref,
        "ocr_text": frame.ocr_text,
    }
    if frame.confidence is not None:
        d["confidence"] = frame.confidence
    if frame.blocks is not None:
        d["blocks"] = frame.blocks
    return d


def _asset_to_dict(a: AssetRef) -> dict:
    return {"key": a.key, "sha256": a.sha256, "size": a.size, "media_type": a.media_type}


def to_dict(result: ExtractionResult) -> dict:
    """Ordered plain-dict form of an :class:`ExtractionResult`.

    Inclusion policy (pinned):

    * ``kind``, ``text``, ``language``, ``assets``, ``meta`` are ALWAYS present
      (``language`` may be ``null``; ``assets`` may be ``[]``).
    * A modality list (``segments``/``frames``/``cards``) that is ``None`` —
      i.e. not applicable to this kind — is OMITTED. One that applies but is
      empty serializes as ``[]``.
    """
    d: dict = {"kind": result.kind, "text": result.text, "language": result.language}
    if result.segments is not None:
        d["segments"] = [_segment_to_dict(s) for s in result.segments]
    if result.frames is not None:
        d["frames"] = [_frame_to_dict(f) for f in result.frames]
    if result.cards is not None:
        d["cards"] = [_card_to_dict(c) for c in result.cards]
    d["assets"] = [_asset_to_dict(a) for a in result.assets]
    d["meta"] = result.meta
    return d


def serialize(result: ExtractionResult) -> str:
    """Serialize to the canonical JSON bytes (UTF-8 text).

    The same bytes back the ``/extractions/{id}/result`` route and the durable
    ``result.json`` bundle member, so this is an operational contract even
    though DailyNotes hashes ``text`` + committed artifacts, not this JSON.

    Float formatting is deliberately Python's ``json``/``repr`` shortest
    round-trip encoding — it is platform-independent (IEEE-754 shortest repr),
    so it is deterministic across the macOS client and Linux server. We pin it by
    relying on this one encoder everywhere (never FastAPI/Pydantic serialization).
    """
    return json.dumps(to_dict(result), indent=2, ensure_ascii=False)
