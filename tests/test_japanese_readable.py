"""Character alignment must not impose English word limits on Japanese."""
from transcript.readable import readable_transcript
from transcript.types import Segment, Transcript, Word


def aligned(text, start=0, speaker='A', step=.04):
    words = [Word(c, round(start+i*step, 4), round(start+(i+1)*step, 4),
                  speaker=speaker) for i, c in enumerate(text)]
    return Segment(text, words[0].start, words[-1].end, speaker, words)


def test_japanese_real_fragments_and_short_exchanges():
    opening = aligned('今日は、沖縄の竹富島に行ったときのブログをみんなにシェアしたいと思います。')
    pieces = [opening]
    start = 3.
    for text, label in [('カ','A'), ('ー','B'), ('ドで。','A'), ('は','A'), ('い。',None),
                        ('はい。','B'), ('いいえ。','A'), ('すい',None), ('ません。','A')]:
        piece = aligned(text, start, label)
        pieces.append(piece)
        start = piece.end
    source = Transcript(pieces, 'ja')
    result = readable_transcript(source)
    assert [s.text for s in result.segments] == [opening.text, 'カードで。', 'はい。',
                                               'はい。', 'いいえ。', 'すいません。']
    assert result.segments[1].speaker is None
    assert ''.join(s.text for s in result.segments) == ''.join(s.text for s in pieces)
    assert [w for s in result.segments for w in s.words] == [w for s in pieces for w in s.words]
    assert all(s.start == s.words[0].start and s.end == s.words[-1].end for s in result.segments)
    assert all(j['basis'] == 'japanese_phrase_continuity_heuristic'
               for j in result.meta['readable']['sentence_continuity_joins'])


def test_japanese_pauses_missing_timing_and_complete_replies_stay_separate():
    for pieces in ([aligned('カ'), aligned('ード。', 1, 'B')],
                   [aligned('はい'), aligned('ええ', .08, 'B')],
                   [aligned('カ', step=.4), aligned('ード。', .4, 'B', step=.4)]):
        assert len(readable_transcript(Transcript(pieces, 'ja')).segments) == 2
    left, right = aligned('カ'), aligned('ード。', .04, 'B')
    right.words[0].start = None
    result = readable_transcript(Transcript([left, right], 'ja'))
    assert len(result.segments) == 2
