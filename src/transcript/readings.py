"""Readings for the whole-sentence view: kana over Japanese kanji, pinyin over Chinese.

Japanese rows are read with MeCab and UniDic (``fugashi`` + ``unidic-lite``), so
an inflected word reads in context (行った → 行っ いっ + た) where a reading of
the bare kanji would guess. Chinese rows are read with ``pypinyin``, whose
phrase dictionary settles polyphones (银行 → yín háng). Readings are spans of
the row's text (``char_start``/``char_end`` like the words' offsets), not a
field per word: the aligner times Japanese one character at a time, so a word
is often a fragment of the unit that has a reading.

Both libraries are optional (``server`` extra) and imported lazily; when one
is missing the rows carry no readings and ``meta.sentences.readings`` reports
why, so clients fall back to their own tokenizer.
"""

from __future__ import annotations

from functools import lru_cache

from .engine import _pkg_version


def _is_kanji(ch: str) -> bool:
    return ("一" <= ch <= "鿿" or "㐀" <= ch <= "䶿" or "豈" <= ch <= "﫿"
            or "\U00020000" <= ch <= "\U0003134f" or ch in "々〆ヶ")


def _hiragana(kana: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in kana)


@lru_cache(maxsize=1)
def _tagger():
    import fugashi
    return fugashi.Tagger()


def japanese_readings(text: str) -> list[dict]:
    """Each kanji-bearing morpheme's reading in hiragana, as a span of ``text``.

    The reading covers the whole morpheme (行っ → いっ); placing it over the
    kanji alone is the client's choice. Morphemes UniDic has no reading for are
    left out rather than guessed.
    """
    spans, cursor = [], 0
    for token in _tagger()(text):
        surface = token.surface
        position = text.find(surface, cursor) if surface else -1
        if position < 0:
            continue
        cursor = position + len(surface)
        if not any(_is_kanji(c) for c in surface):
            continue
        kana = getattr(token.feature, "kana", None)
        if not kana or kana == "*":
            continue
        reading = _hiragana(kana)
        if reading != surface:
            spans.append({"char_start": position, "char_end": cursor, "reading": reading})
    return spans


def chinese_readings(text: str) -> list[dict]:
    """Each hanzi's pinyin with tone marks, read in its phrase, as a span of ``text``."""
    from pypinyin import Style, pinyin

    syllables = pinyin(text, style=Style.TONE, errors=lambda chars: [""] * len(chars))
    if len(syllables) != len(text):
        return []  # never misplace a reading
    return [{"char_start": i, "char_end": i + 1, "reading": choices[0]}
            for i, (ch, choices) in enumerate(zip(text, syllables))
            if _is_kanji(ch) and choices and choices[0]]


_READERS = {
    "ja": (japanese_readings, "unidic-lite", ("fugashi", "unidic-lite")),
    "zh": (chinese_readings, "pypinyin", ("pypinyin",)),
}


def attach_readings(data: dict, language: str | None) -> dict:
    """Add ``readings`` to each row of a ``unit=sentence`` JSON dict, in place.

    Only Japanese and Chinese rows are read; other languages are returned as
    they are. ``meta.sentences.readings`` names the source and its versions,
    or the error that left the rows without readings.
    """
    reader = _READERS.get((language or "").split("-")[0].lower())
    if reader is None:
        return data
    read, source, packages = reader
    info = {"source": source, "versions": {p: _pkg_version(p) for p in packages}, "error": None}
    try:
        rows = [read(segment["text"]) for segment in data["segments"]]
    except Exception as exc:  # optional dependency missing or broken dictionary
        info["error"] = f"{type(exc).__name__}: {exc}"
    else:
        for segment, readings in zip(data["segments"], rows):
            segment["readings"] = readings
    data.setdefault("meta", {}).setdefault("sentences", {})["readings"] = info
    return data
