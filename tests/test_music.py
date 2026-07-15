"""Unit tests for music tagging (pure overlap logic — no inaSpeechSegmenter)."""

from transcript.music import detect_and_tag, tag_music
from transcript.types import Segment, Transcript


def _t(*segs):
    return Transcript(segments=list(segs))


def test_flags_segment_mostly_inside_music():
    t = _t(Segment(text="la la la", start=10.0, end=14.0))
    assert tag_music(t, [(9.0, 13.5)]) == 1  # 3.5s of 4s inside music
    assert t.segments[0].music is True


def test_skips_segment_with_minor_overlap():
    t = _t(Segment(text="hello", start=0.0, end=10.0))
    assert tag_music(t, [(9.0, 12.0)]) == 0  # only 1s of 10s
    assert t.segments[0].music is False


def test_sums_overlap_across_multiple_ranges():
    t = _t(Segment(text="medley", start=0.0, end=10.0))
    # 3s + 3s = 6s of 10s >= 50%
    assert tag_music(t, [(0.0, 3.0), (5.0, 8.0)]) == 1


def test_ignores_segments_without_timing():
    t = _t(Segment(text="no times"), Segment(text="bad", start=5.0, end=5.0))
    assert tag_music(t, [(0.0, 100.0)]) == 0


def test_detect_and_tag_survives_missing_dependency():
    # In the dev env inaSpeechSegmenter isn't installed; the pipeline must not fail.
    t = _t(Segment(text="hi", start=0.0, end=1.0))
    detect_and_tag(t, "/nonexistent.wav")
    assert t.segments[0].music is False
