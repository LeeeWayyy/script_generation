from fastapi.testclient import TestClient

from transcript.readable import _phrase_starts, sentence_transcript
from transcript.types import Segment, Transcript, Word


def timed_words(text, start=0.0, step=.5, speaker='A'):
    return [Word(w, start + i * step, start + i * step + .4, speaker=speaker)
            for i, w in enumerate(text.split())]


def test_sentence_spans_segments_and_mixed_speaker_stays_whole():
    first = timed_words('I think we', 0)
    second = timed_words('should go. Yes.', 2, speaker='B')
    result = sentence_transcript(Transcript([
        Segment('I think we', 0, 1.4, 'A', first),
        Segment('should go. Yes.', 2, 3.4, 'B', second),
    ], 'en'))
    assert [s.text for s in result.segments] == ['I think we should go.', 'Yes.']
    assert result.segments[0].speaker is None
    assert [w.speaker for w in result.segments[0].words] == ['A', 'A', 'A', 'B', 'B']
    assert (result.segments[0].start, result.segments[0].end) == (0, 2.9)
    assert result.segments[1].speaker == 'B'
    words = result.to_dict(offsets=True)['segments'][0]['words']
    assert [(w['char_start'], w['char_end']) for w in words] == [(0, 1), (2, 7), (8, 10), (11, 17), (18, 21)]
    assert result.meta['sentences']['forced_boundaries'] == []


def test_long_unpunctuated_run_splits_between_words_at_longest_pause():
    words = timed_words('one two three four five six seven eight', step=5)
    words[5].start += 1.5  # longest pause, before "six"
    words[5].end += 1.5
    words[2].word = 'three,'  # a comma alone doesn't beat a 1.5 s longer pause
    text = ' '.join(w.word for w in words)
    result = sentence_transcript(Transcript([Segment(text, 0, 36, words=words)], 'en'))
    assert [s.text for s in result.segments] == ['one two three, four five', 'six seven eight']
    assert [w for s in result.segments for w in s.words] == words
    assert result.meta['sentences']['forced_boundaries'] == [0]


def test_japanese_split_waits_for_a_phrase_boundary():
    text = '今日は天気です'
    phrase = min(p for p in _phrase_starts(text) if p > 0)
    words = [Word(c, i * 6.0, i * 6.0 + 5) for i, c in enumerate(text)]
    for later in words[phrase + 1:]:  # biggest pause sits inside the next phrase
        later.start += 3
        later.end += 3
    result = sentence_transcript(Transcript([Segment(text, 0, 45, words=words)], 'ja'))
    assert [s.text for s in result.segments] == [text[:phrase], text[phrase:]]


def test_ellipsis_continues_but_period_before_lowercase_ends():
    words = timed_words('so... we left. then home.')
    result = sentence_transcript(Transcript([Segment('so... we left. then home.', 0, 3, words=words)], 'en'))
    assert [s.text for s in result.segments] == ['so... we left.', 'then home.']


def test_unmapped_segment_passes_through_and_is_reported():
    words = timed_words('Hi there.')
    source = [Segment('Hi there.', 0, 1, 'A', words), Segment('No words here', 2, 3, 'A'),
              Segment('Bye.', 4, 5, 'A', [Word('Hey', 4, 5, speaker='A')])]
    result = sentence_transcript(Transcript(source, 'en'))
    assert [s.text for s in result.segments] == ['Hi there.', 'No words here', 'Bye.']
    assert result.segments[1] == source[1]
    assert [(f['segment_index'], f['reason']) for f in result.meta['sentences']['fallbacks']] == [
        (1, 'no_words'), (2, 'words_do_not_match_text')]


def test_result_route_unit_sentence(monkeypatch, tmp_path):
    from transcript.server import Job, create_app
    monkeypatch.delenv('TRANSCRIPT_TOKEN', raising=False)
    monkeypatch.setenv('TRANSCRIPT_DATA_DIR', str(tmp_path))
    app = create_app()
    transcript = Transcript([Segment('Hi', 0, 1, words=timed_words('Hi')),
                             Segment('there.', 1, 2, words=timed_words('there.', 1))], 'en')
    app.state.job_store.add(Job(id='abc123', source='x.wav', status='done', transcript=transcript))
    with TestClient(app) as client:
        path = '/jobs/abc123/result'
        body = client.get(path, params={'format': 'json', 'unit': 'sentence'}).json()
        assert [s['text'] for s in body['segments']] == ['Hi there.']
        assert body['segments'][0]['words'][1]['char_start'] == 3
        for params in ({'unit': 'sentence'}, {'format': 'json', 'unit': 'sentence', 'readable': 'true'},
                       {'format': 'json', 'unit': 'word'}):
            assert client.get(path, params=params).status_code == 400
        assert client.get(path, params={'format': 'json'}).json() == transcript.to_dict()
