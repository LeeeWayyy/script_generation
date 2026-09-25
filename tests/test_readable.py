import pytest
from fastapi.testclient import TestClient

from transcript.readable import readable_transcript
from transcript.types import Segment, Transcript, Word


def test_representative_interview_preserves_text_words_and_turns():
    text = "If you watch videos on Chinese TikTok Is that Weibo? It's Douyin Xiaohongshu. Yeah "
    text += "i could roll like that you have such a good accent oh thank you very much " * 4
    tokens = text.split()
    words = [Word(token, i * .4, (i + 1) * .4, speaker=(
        "SPEAKER_00" if i < 8 or i >= 12 else "SPEAKER_01"
    )) for i, token in enumerate(tokens)]
    source = Transcript([Segment(text.strip(), 0, len(words) * .4, words=words)])
    result = readable_transcript(source)
    assert " ".join(s.text for s in result.segments) == text.strip()
    assert [w for s in result.segments for w in s.words] == words
    assert len(result.segments) > 5
    for segment in result.segments:
        assert len(segment.words) <= 20
        assert segment.end - segment.start <= 8.01
        assert {w.speaker for w in segment.words} == {segment.speaker}
        assert (segment.start, segment.end) == (segment.words[0].start, segment.words[-1].end)
    assert result.meta['readable']['fallbacks'] == []
    assert len(source.segments) == 1 and source.meta == {}


@pytest.mark.parametrize('mode', ['no_words', 'missing_time', 'mismatch', 'no_speakers'])
def test_explicit_fallbacks(mode):
    text = ' '.join(['spoken'] * 45)
    words = [] if mode == 'no_words' else [Word('spoken', i, i + .5) for i in range(45)]
    if mode == 'missing_time':
        words[0].start = None
    if mode == 'mismatch':
        words[0].word = 'different'
    result = readable_transcript(Transcript([Segment(text, 0, 45, words=words)]))
    assert ' '.join(s.text for s in result.segments) == text
    assert all(s.speaker is None for s in result.segments)
    assert result.meta['readable']['fallbacks']
    if mode == 'mismatch':
        assert result.meta['readable']['fallbacks'][0]['source_words'][0]['word'] == 'different'
    if mode != 'no_speakers':
        assert result.segments[0].start is None
        assert result.segments[0].end is None


def test_single_speaker_caption_and_pause():
    caption = Segment('Hello everyone', 3, 5, 'SPEAKER_00')
    result = readable_transcript(Transcript([caption]))
    assert result.segments == [caption]
    assert result.meta['readable']['fallbacks'][0]['timing'] == 'source_segment'
    words = [Word('Hello', 0, .5, speaker='SPEAKER_00'),
             Word('again', 2, 3, speaker='SPEAKER_00')]
    result = readable_transcript(Transcript([Segment('Hello again', 0, 3, words=words)]))
    assert [s.text for s in result.segments] == ['Hello', 'again']


def test_result_opt_in_keeps_legacy_and_cached_result(monkeypatch, tmp_path):
    from transcript.server import Job, create_app
    monkeypatch.delenv('TRANSCRIPT_TOKEN', raising=False)
    monkeypatch.setenv('TRANSCRIPT_DATA_DIR', str(tmp_path))
    app = create_app()
    transcript = Transcript([Segment(' '.join(['hello'] * 45), 0, 30)])
    job = Job(id='abc123', source='example.wav', status='done', transcript=transcript)
    app.state.job_store.add(job)
    with TestClient(app) as client:
        path = '/jobs/abc123/result'
        assert client.get(path, params={'format': 'json'}).json() == transcript.to_dict()
        response = client.get(path, params={'format': 'json', 'readable': 'true'})
        assert response.status_code == 200
        assert len(response.json()['segments']) == 3
        assert response.json()['segments'][0]['start'] is None
        assert client.get(path, params={'readable': 'true'}).status_code == 400
        assert client.get(path, params={'format': 'json'}).json() == transcript.to_dict()


def test_failed_alignment_metadata_and_unknown_word_speaker_are_honest():
    words = [Word('One.', 0, 1, speaker='SPEAKER_00'), Word('Two.', None, None)]
    source = Transcript([Segment('One. Two.', 0, 4, 'SPEAKER_00', words)], meta={
        'align_requested': True, 'align_succeeded': False, 'diarize_requested': True,
    })
    result = readable_transcript(source)
    assert result.meta['align_succeeded'] is False
    assert result.segments[1].speaker is None
    assert result.segments[1].start is None and result.segments[1].end is None
    assert result.meta['readable']['fallbacks'][0]['source_end'] == 4
