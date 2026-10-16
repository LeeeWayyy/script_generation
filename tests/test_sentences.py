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
    result = sentence_transcript(Transcript([Segment(text, 0, 36, words=words)], 'en'),
                                 predict=lambda texts: [None] * len(texts))
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
    result = sentence_transcript(Transcript([Segment(text, 0, 45, words=words)], 'ja'),
                                 predict=lambda texts: [None] * len(texts))
    assert [s.text for s in result.segments] == [text[:phrase], text[phrase:]]


def test_ellipsis_continues_but_period_before_lowercase_ends():
    words = timed_words('so... we left. then home.')
    result = sentence_transcript(Transcript([Segment('so... we left. then home.', 0, 3, words=words)], 'en'))
    assert [s.text for s in result.segments] == ['so... we left.', 'then home.']



def test_ascii_closing_quote_ends_a_sentence():
    text = "He said 'go.' Then left."
    result = sentence_transcript(Transcript([Segment(text, 0, 3, words=timed_words(text))], 'en'))
    assert [s.text for s in result.segments] == ["He said 'go.'", 'Then left.']


def test_continuing_ellipses_do_not_count_as_punctuation():
    text = 'so... we ' + ' '.join(['word'] * 118)
    words = timed_words(text)  # 60 s with one "stop", an ellipsis that continues
    calls = []
    sentence_transcript(Transcript([Segment(text, 0, 60, words=words)], 'en'),
                        predict=lambda texts: calls.append(texts) or [None] * len(texts))
    assert calls

def test_unmapped_segment_passes_through_and_is_reported():
    words = timed_words('Hi there.')
    source = [Segment('Hi there.', 0, 1, 'A', words), Segment('No words here', 2, 3, 'A'),
              Segment('Bye.', 4, 5, 'A', [Word('Hey', 4, 5, speaker='A')])]
    result = sentence_transcript(Transcript(source, 'en'))
    assert [s.text for s in result.segments] == ['Hi there.', 'No words here', 'Bye.']
    assert result.segments[1] == source[1]
    assert [(f['segment_index'], f['reason']) for f in result.meta['sentences']['fallbacks']] == [
        (1, 'no_words'), (2, 'words_do_not_match_text')]



def test_segments_join_by_the_scripts_they_meet():
    def joined(language, *texts):
        segments, clock = [], 0.0
        for text in texts:
            segments.append(Segment(text, clock, clock + 1, words=[Word(text, clock, clock + 1)]))
            clock += 1
        return [s.text for s in sentence_transcript(Transcript(segments, language)).segments]
    assert joined('ja', '今日は', '晴れ。') == ['今日は晴れ。']
    assert joined('ja', 'Hello', 'world.') == ['Hello world.']
    assert joined('ja', 'これは', 'iPhone。') == ['これはiPhone。']
    assert joined('th', 'สวัสดี', 'ครับ.') == ['สวัสดีครับ.']
    assert joined('en', 'Hello', 'world.') == ['Hello world.']


def test_untimed_edge_words_widen_sentence_bounds_or_null_them():
    def sentence(*words, source_end=None, then=None):
        segments = [Segment(' '.join(w.word for w in words), 0, source_end, words=list(words))]
        if then is not None:
            segments.append(Segment('Next.', then, then + 1, words=[Word('Next.', then, then + 1)]))
        return sentence_transcript(Transcript(segments, 'en'))
    result = sentence(Word('One', 0, 1), Word('two.'), then=3)
    assert (result.segments[0].start, result.segments[0].end) == (0, 3)
    assert result.meta['sentences']['estimated_bounds'] == [0]
    result = sentence(Word('One', 0, 1), Word('two.'), source_end=2, then=3)
    assert (result.segments[0].start, result.segments[0].end) == (0, 2)
    result = sentence(Word('One', 0, 1), Word('two.'))
    assert result.segments[0].end is None
    result = sentence(Word('One', 1, 1.5), Word('two.', .5, 2))  # out of order
    assert (result.segments[0].start, result.segments[0].end) == (.5, 2)
    assert result.meta['sentences']['estimated_bounds'] == []

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


def test_model_boundaries_need_a_pause_and_sparse_punctuation():
    text = 'はいそうですね次に行きます'
    gaps = {2: 1.0, 7: 0.0}  # model proposes both; only the paused one is used
    words, clock = [], 0.0
    for i, c in enumerate(text):
        clock += gaps.get(i, 0)
        words.append(Word(c, clock, clock + 1))
        clock += 1
    calls = []

    def predict(texts):
        calls.append(texts)
        return [{2, 7}]
    result = sentence_transcript(Transcript([Segment(text, 0, clock, words=words)], 'ja'),
                                 predict=predict)
    assert calls == [[text]]
    assert [s.text for s in result.segments] == ['はい', 'そうですね次に行きます']
    meta = result.meta['sentences']
    assert meta['model_boundaries'] == [0] and meta['punctuation_model']['used']
    punctuated = ('ok. ' * 30).strip()
    words = timed_words(punctuated)
    sentence_transcript(Transcript([Segment(punctuated, 0, 15, words=words)], 'en'), predict=predict)
    assert len(calls) == 1  # a full stop every half second: the model is never asked


def test_model_failure_is_reported_not_raised():
    def broken(texts):
        raise ImportError('punctuators missing')
    words = timed_words('one two three four five six seven eight', step=2)
    result = sentence_transcript(Transcript([Segment('one two three four five six seven eight', 0, 16,
                                                     words=words)], 'en'), predict=broken)
    assert [s.text for s in result.segments] == ['one two three four five six seven eight']
    assert result.meta['sentences']['punctuation_model']['error'] == 'ImportError: punctuators missing'


def test_predicted_starts_map_back_through_rewritten_text(monkeypatch):
    from transcript import readable

    class Model:
        def infer(self, texts, apply_sbd):
            return [['水素を2倍入れた。', 'ねー。', 'でも、見えた。'], ['changed words.', 'x']]
    monkeypatch.setattr(readable, '_punctuation_model', lambda: Model())
    text = '水素を２倍入れたねーでも見えた'
    assert readable._predicted_sentence_starts([text, 'other words x']) == [
        {text.index('ね'), text.index('で')}, None]
