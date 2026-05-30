"""Tests for the formatters and data model — no ML deps required."""

import json

from transcript.formats import to_srt, to_txt, to_vtt, to_json, render
from transcript.types import Segment, Transcript, Word


def make_transcript() -> Transcript:
    return Transcript(
        language="en",
        segments=[
            Segment(
                text="Hello there.",
                start=0.0,
                end=1.5,
                speaker="SPEAKER_00",
                words=[Word("Hello", 0.0, 0.5, 0.9, "SPEAKER_00")],
            ),
            Segment(text="General Kenobi.", start=1.6, end=3.2, speaker="SPEAKER_01"),
        ],
    )


def test_txt_includes_speaker_labels():
    out = to_txt(make_transcript())
    assert "SPEAKER_00: Hello there." in out
    assert "SPEAKER_01: General Kenobi." in out


def test_txt_without_speakers():
    t = Transcript(segments=[Segment(text="Just text.", start=0, end=1)])
    assert to_txt(t).strip() == "Just text."


def test_srt_timestamps_use_comma():
    out = to_srt(make_transcript())
    assert "00:00:00,000 --> 00:00:01,500" in out
    assert "[SPEAKER_00] Hello there." in out
    # First cue is numbered 1.
    assert out.startswith("1\n")


def test_vtt_header_and_dot_timestamps():
    out = to_vtt(make_transcript())
    assert out.startswith("WEBVTT")
    assert "00:00:01.600 --> 00:00:03.200" in out


def test_json_roundtrips():
    out = to_json(make_transcript())
    data = json.loads(out)
    assert data["language"] == "en"
    assert data["segments"][0]["speaker"] == "SPEAKER_00"
    assert data["segments"][0]["words"][0]["word"] == "Hello"


def test_render_dispatch_and_speakers_property():
    t = make_transcript()
    assert render(t, "txt") == to_txt(t)
    assert t.speakers == ["SPEAKER_00", "SPEAKER_01"]
    assert t.text == "Hello there.\nGeneral Kenobi."
