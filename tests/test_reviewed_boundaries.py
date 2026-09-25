"""Human-adjudicated sentence repairs, not automatic speaker smoothing."""
from dataclasses import replace

import pytest

from transcript.readable import join_reviewed_boundaries
from transcript.types import Segment, Transcript, Word


def interview():
    # Actual words/times from job 5451f7c29851; scores are ASR, not speaker confidence.
    rows = [
        [('I', 93.760, 93.840, '01'), ('made', 93.860, 94.020, '01')],
        [('it', 94.060, 94.100, '00'), ('up.', 94.200, 94.280, '00')],
        [('No,', 94.640, 94.821, '03')],
        [('Alice,', 98.442, 98.723, '02'), ('go', 98.743, 98.903, '02')],
        [('ahead.', 98.943, 99.183, '01')],
        [('Tell', 99.223, 99.403, '01'), ('him.', 99.423, 99.523, '01')],
    ]
    segments = []
    for row in rows:
        words = [Word(text, start, end, .99, 'SPEAKER_' + speaker)
                 for text, start, end, speaker in row]
        segments.append(Segment(' '.join(w.word for w in words), words[0].start,
                                words[-1].end, words[0].speaker, words))
    return Transcript(segments, 'en', {'readable': {'version': 1, 'fallbacks': []}})


def test_only_reviewed_boundaries_change_preserving_words_timing_and_uncertainty():
    source = interview()
    original = source.to_dict()
    result = join_reviewed_boundaries(source, [94.060, 98.943])
    assert [(s.text, s.start, s.end) for s in result.segments] == [
        ('I made it up.', 93.760, 94.280), ('No,', 94.640, 94.821),
        ('Alice, go ahead.', 98.442, 99.183), ('Tell him.', 99.223, 99.523),
    ]
    assert result.segments[0].speaker is None and result.segments[2].speaker is None
    assert [w for s in result.segments for w in s.words] == [w for s in source.segments for w in s.words]
    assert source.to_dict() == original
    assert [f['segment_index'] for f in result.meta['readable']['fallbacks']] == [0, 2]
    assert all(f['speaker'] == 'unavailable' for f in result.meta['readable']['fallbacks'])
    assert result.segments[1] == source.segments[2]
    assert result.segments[3] == source.segments[5]
    assert join_reviewed_boundaries(source, []).segments == source.segments


def test_existing_fallbacks_follow_shifted_rows_and_repeated_repairs():
    source = interview()
    source.meta['readable']['fallbacks'] = [{'segment_index': 5, 'speaker': 'unavailable'}]
    first = join_reviewed_boundaries(source, [98.943])
    result = join_reviewed_boundaries(first, [94.060])
    assert result.meta['readable']['fallbacks'][0]['segment_index'] == 3
    assert [c['segment_index'] for c in result.meta['readable']['reviewed_joins']] == [2, 0]


@pytest.mark.parametrize('times', [[float('nan')], [float('inf')], [-1], [True],
                                   [94.060, 94.060], [93.760], [94.061]])
def test_invalid_or_stale_boundary_rejected(times):
    with pytest.raises(ValueError):
        join_reviewed_boundaries(interview(), times)


@pytest.mark.parametrize('broken', ['missing', 'overlap', 'nan', 'coarse'])
def test_no_fabricated_timing(broken):
    source = interview()
    right = source.segments[1]
    if broken == 'missing':
        right.words[0] = replace(right.words[0], start=None)
    elif broken == 'overlap':
        source.segments[0].words[-1] = replace(source.segments[0].words[-1], end=94.08)
        source.segments[0].end = 94.08
    elif broken == 'nan':
        right.words[0] = replace(right.words[0], end=float('nan'))
    else:
        right.end = 95
    with pytest.raises(ValueError, match='timing'):
        join_reviewed_boundaries(source, [94.060])


def test_true_short_exchanges_preserved_without_explicit_review():
    source = Transcript([
        Segment('Go', 0, .2, 'A', [Word('Go', 0, .2, .01, 'A')]),
        Segment('no', .24, .4, 'B', [Word('no', .24, .4, 1., 'B')]),
        Segment('okay', .44, .7, 'A', [Word('okay', .44, .7, .99, 'A')]),
    ])
    assert join_reviewed_boundaries(source, []).segments == source.segments


def test_automatic_sentence_continuity_without_review_or_text_rewriting():
    from transcript.readable import readable_transcript
    source = interview()
    result = readable_transcript(source)
    assert [s.text for s in result.segments] == [
        'I made it up.', 'No,', 'Alice, go', 'ahead.', 'Tell him.',
    ]
    # The artificial standalone "No," has no sentence end, so the following
    # phrase is deliberately not eligible. Real interview "No, ... saying."
    # is sentence-terminated, as checked below.
    source.segments[2] = replace(source.segments[2], text='No.',
                                 words=[replace(source.segments[2].words[0], word='No.')])
    result = readable_transcript(source)
    assert [s.text for s in result.segments] == [
        'I made it up.', 'No.', 'Alice, go ahead.', 'Tell him.',
    ]
    assert result.segments[0].start == 93.76 and result.segments[0].end == 94.28
    assert result.segments[2].start == 98.442 and result.segments[2].end == 99.183
    assert result.segments[0].speaker is None
    assert 'reviewed_joins' not in result.meta['readable']
    assert all(c['basis'] == 'short_sentence_continuity_heuristic'
               for c in result.meta['readable']['sentence_continuity_joins'])
    assert [w for s in result.segments for w in s.words] == [w for s in source.segments for w in s.words]


@pytest.mark.parametrize('left,right,gap,right_duration,language', [
    ('Go', 'no.', .04, .10, 'en'),  # fast, real interjection
    ('I agree', 'yeah.', .04, .10, 'en'),
    ('Hello.', 'there.', .04, .20, 'en'),  # completed sentence
    ('I made', 'it up.', .20, .22, 'en'),  # pause
    ('I made', 'it up.', -.02, .22, 'en'),  # overlapping speech
    ('I made', 'it up.', .04, .50, 'en'),  # sustained second turn
    ('I made', 'It up.', .04, .22, 'en'),  # capitalized new utterance
    ('I made,', 'it up.', .04, .22, 'en'),  # explicit clause boundary
    ('I made', 'it up?', .04, .22, 'en'),  # question/echo
    ('I made', 'it up.', .04, .22, 'es'),  # unsupported language
])
def test_automatic_rule_preserves_rapid_turns_and_uncertain_context(
        left, right, gap, right_duration, language):
    from transcript.readable import readable_transcript
    start = .26 + gap
    source = Transcript([
        Segment(left, 0, .26, 'A', [Word(left, 0, .26, .01, 'A')]),
        Segment(right, start, start + right_duration, 'B',
                [Word(right, start, start + right_duration, 1., 'B')]),
    ], language)
    result = readable_transcript(source)
    assert result.segments == source.segments
    assert not result.meta['readable'].get('sentence_continuity_joins')


def test_automatic_rule_does_not_promote_missing_timing_or_unknown_speakers():
    from transcript.readable import readable_transcript
    for field, value in [('start', None), ('speaker', None)]:
        source = interview()
        source.segments[1].words[0] = replace(source.segments[1].words[0], **{field: value})
        result = readable_transcript(source)
        assert not any(s.text == 'I made it up.' for s in result.segments)
