import pytest
from fastapi.testclient import TestClient

from transcript import readings
from transcript.readings import attach_readings, chinese_readings, japanese_readings
from transcript.types import Segment, Transcript, Word


def test_japanese_readings_follow_inflection_and_skip_kana():
    pytest.importorskip('fugashi')
    pytest.importorskip('unidic_lite')
    text = '行ったけど、東京は暑かった。'
    spans = japanese_readings(text)
    assert [(text[s['char_start']:s['char_end']], s['reading']) for s in spans] == [
        ('行っ', 'いっ'), ('東京', 'とうきょう'), ('暑かっ', 'あつかっ')]


def test_japanese_spans_map_back_and_unread_tokens_are_left_out(monkeypatch):
    class Feature:
        def __init__(self, kana):
            self.kana = kana

    class Token:
        def __init__(self, surface, kana):
            self.surface, self.feature = surface, Feature(kana)
    tokens = [Token('今日', 'キョウ'), Token('は', 'ハ'), Token('㐂', '*'), Token('雨', 'アメ')]
    monkeypatch.setattr(readings, '_tagger', lambda: lambda text: tokens)
    text = '今日は 㐂雨'
    assert japanese_readings(text) == [
        {'char_start': 0, 'char_end': 2, 'reading': 'きょう'},
        {'char_start': 5, 'char_end': 6, 'reading': 'あめ'}]


def test_chinese_readings_settle_polyphones_per_hanzi():
    pytest.importorskip('pypinyin')
    text = '我用iPhone去银行还钱。'
    spans = chinese_readings(text)
    assert [(text[s['char_start']], s['reading']) for s in spans] == [
        ('我', 'wǒ'), ('用', 'yòng'), ('去', 'qù'), ('银', 'yín'), ('行', 'háng'),
        ('还', 'huán'), ('钱', 'qián')]
    assert [s['reading'] for s in chinese_readings('二〇二六年')] == [
        'èr', 'líng', 'èr', 'liù', 'nián']


def test_missing_library_is_reported_and_other_languages_untouched(monkeypatch):
    def missing(text):
        raise ImportError('No module named fugashi')
    monkeypatch.setitem(readings._READERS, 'ja', (missing, 'unidic-lite', ('fugashi',)))
    data = {'segments': [{'text': '雨'}], 'meta': {'sentences': {}}}
    attach_readings(data, 'ja')
    assert 'readings' not in data['segments'][0]
    assert data['meta']['sentences']['readings']['error'] == 'ImportError: No module named fugashi'
    english = {'segments': [{'text': 'Hi.'}], 'meta': {}}
    assert attach_readings(english, 'en') == {'segments': [{'text': 'Hi.'}], 'meta': {}}


def test_sentence_route_carries_readings(monkeypatch, tmp_path):
    from transcript.server import Job, create_app
    monkeypatch.delenv('TRANSCRIPT_TOKEN', raising=False)
    monkeypatch.setenv('TRANSCRIPT_DATA_DIR', str(tmp_path))
    monkeypatch.setitem(readings._READERS, 'ja', (
        lambda text: [{'char_start': 0, 'char_end': 1, 'reading': 'あめ'}], 'unidic-lite', ()))
    app = create_app()
    transcript = Transcript([Segment('雨。', 0, 1, words=[Word('雨。', 0, 1)])], 'ja')
    app.state.job_store.add(Job(id='abc123', source='x.wav', status='done', transcript=transcript))
    with TestClient(app) as client:
        body = client.get('/jobs/abc123/result', params={'format': 'json', 'unit': 'sentence'}).json()
    assert body['segments'][0]['readings'] == [{'char_start': 0, 'char_end': 1, 'reading': 'あめ'}]
    assert body['meta']['sentences']['readings']['error'] is None
    assert 'readings' not in transcript.meta.get('sentences', {})
