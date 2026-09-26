from transcript.engine import _trim_sentence_tails


def test_alignment_boundary_duplicate_is_removed_only_with_exact_evidence():
    from copy import deepcopy
    from transcript.engine import _deduplicate_alignment_boundaries, _to_transcript
    from transcript.readable import readable_transcript

    words = [{'word': c, 'start': i / 10, 'end': (i + 1) / 10,
              'score': .9, 'speaker': 'A'} for i, c in enumerate('えぇ!?')]
    original = {'segments': [
        {'text': 'えぇ!', 'start': 0., 'end': .4, 'speaker': 'A', 'words': words},
        {'text': '?', 'start': .3, 'end': .4, 'speaker': 'A', 'words': [dict(words[-1])]},
    ], 'word_segments': words + [dict(words[-1])]}
    data = deepcopy(original)
    changes = _deduplicate_alignment_boundaries(data)
    assert len(changes) == 1 and changes[0]['duplicate_word'] == words[-1]
    assert data['segments'][0]['end'] == .3
    assert data['word_segments'] == words
    raw = _to_transcript(data, language='en')  # no ML/language-model dependency
    readable = readable_transcript(raw)
    assert [w for s in readable.segments for w in s.words] == [w for s in raw.segments for w in s.words]
    assert ''.join(s.text for s in readable.segments) == 'えぇ!?'
    assert _deduplicate_alignment_boundaries(data) == []

    for mutation in ('covered_by_text', 'different_evidence', 'overlap'):
        data = deepcopy(original)
        if mutation == 'covered_by_text':
            data['segments'][0]['text'] += '?'
        elif mutation == 'different_evidence':
            data['segments'][1]['words'][0]['score'] = .8
        else:
            data['segments'][0]['words'][-2]['end'] = .35
        before = deepcopy(data)
        assert _deduplicate_alignment_boundaries(data) == []
        assert data == before


def test_empty_asr_refuses_output_even_without_diarization(monkeypatch):
    import sys
    from types import SimpleNamespace
    import pytest
    from transcript.engine import TranscriptionEngine

    monkeypatch.setitem(sys.modules, 'whisperx', SimpleNamespace(load_audio=lambda _: []))
    engine = object.__new__(TranscriptionEngine)
    engine.batch_size = 16
    engine._load_asr = lambda: SimpleNamespace(transcribe=lambda *a, **k: {'segments': []})
    with pytest.raises(RuntimeError, match='No reliable speech activity'):
        engine.run('music.wav', diarize=False, language='ja')


def test_no_speech_does_not_publish_empty_or_hallucinated_captions(monkeypatch):
    import sys
    from types import SimpleNamespace
    import pytest
    from transcript.engine import TranscriptionEngine

    monkeypatch.setitem(sys.modules, 'whisperx', SimpleNamespace())
    engine = object.__new__(TranscriptionEngine)
    engine._load_diarizer = lambda: lambda *args, **kwargs: {'start': [], 'end': []}
    for segments in ([], [{'text': 'Thank you.', 'start': 100., 'end': 101.}]):
        with pytest.raises(RuntimeError, match='No reliable speech activity'):
            engine._align_and_diarize([], {'segments': segments}, language='en',
                                     diarize=True, min_speakers=None, max_speakers=None,
                                     align=False)


def test_actual_acoustic_gap_stops_next_turn_leaking_into_final_word():
    # Rounded actual 85–105s acoustic islands, local excerpt time.
    turns = {'start': [8.688, 8.992, 9.177, 9.278, 9.599],
             'end': [8.992, 9.177, 9.278, 9.295, 11.287]}
    word = {'word': 'up.', 'start': 9.194, 'end': 9.794, 'score': .741}
    segment = {'text': 'I made it up.', 'start': 8.754, 'end': 9.794, 'words': [word]}
    adjustments = _trim_sentence_tails({'segments': [segment]}, turns)
    assert word == {'word': 'up.', 'start': 9.194, 'end': 9.295, 'score': .741}
    assert segment['end'] == 9.295
    assert adjustments[0]['original_end'] == 9.794
    assert adjustments[0]['timing_precision'] == 'model_estimate'


def test_no_trim_for_overlap_short_gap_nonterminal_or_missing_timing():
    for word, turns in [
        ({'word': 'God.', 'start': 0., 'end': 1.9},
         {'start': [0., .5], 'end': [.15, 2.]}),  # drawn-out or severely misaligned
        ({'word': 'up.', 'start': 0., 'end': .9},
         {'start': [0., .1], 'end': [.6, 1.]}),  # overlapping speakers cover the gap
        ({'word': 'up.', 'start': 0., 'end': .9},
         {'start': [0., .3], 'end': [.15, 1.]}),  # short stop-consonant gap
        ({'word': 'up', 'start': 0., 'end': .9},
         {'start': [0., .5], 'end': [.15, 1.]}),
        ({'word': 'up.', 'end': .9}, {'start': [0.], 'end': [1.]}),
        ({'word': 'up.', 'start': .4, 'end': .9},
         {'start': [0., .5], 'end': [.15, 1.]}),  # word begins in a gap
        ({'word': 'up.', 'start': 0., 'end': .4},
         {'start': [0., .5], 'end': [.15, 1.]}),  # does not enter next speech
    ]:
        before = dict(word)
        assert _trim_sentence_tails({'segments': [{'words': [word]}]}, turns) == []
        assert word == before


def test_pipeline_adjusts_before_speaker_assignment_and_reports_provenance(monkeypatch):
    import sys
    from types import SimpleNamespace
    from transcript.engine import TranscriptionEngine

    raw = {'segments': [{'text': 'up.', 'start': 9.194, 'end': 9.794,
                         'words': [{'word': 'up.', 'start': 9.194, 'end': 9.794}]}]}
    turns = {'start': [9.177, 9.599], 'end': [9.295, 11.287]}
    def assign(frame, result):
        assert result['segments'][0]['words'][0]['end'] == 9.295
        return result
    monkeypatch.setitem(sys.modules, 'whisperx', SimpleNamespace(assign_word_speakers=assign))
    engine = object.__new__(TranscriptionEngine)
    engine._load_diarizer = lambda: lambda *args, **kwargs: turns
    result = engine._align_and_diarize([], raw, language='en', diarize=True,
                                      min_speakers=None, max_speakers=None, align=False)
    assert result.segments[0].end == 9.295
    assert result.meta['timing_adjustments'][0]['original_end'] == 9.794
