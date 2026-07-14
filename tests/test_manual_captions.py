import json
import subprocess

from transcript.engine import _to_transcript
from transcript.ingest import (
    _manual_caption_language,
    download_manual_caption,
    parse_json3_caption,
)
from transcript.types import Transcript


def test_auto_captions_are_ignored(monkeypatch, tmp_path):
    info = {"id": "video", "automatic_captions": {"en": [{"ext": "json3"}]}}
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(info), stderr="")

    monkeypatch.setattr("transcript.ingest.subprocess.run", fake_run)

    assert download_manual_caption(
        "https://youtube.com/watch?v=video", tmp_path, language="en"
    ) is None
    assert len(calls) == 1


def test_manual_caption_selection_parsing_and_speaker_turn_split(tmp_path):
    info = {
        "language": "es",
        "subtitles": {
            "es": [{"ext": "json3"}],
            "en-US": [{"ext": "json3"}],
        },
        "automatic_captions": {"fr": [{"ext": "json3"}]},
    }
    assert _manual_caption_language(info, None) == "es"
    assert _manual_caption_language(info, "en") == "en-US"
    assert _manual_caption_language(info, "fr") is None

    path = tmp_path / "video.en-US.json3"
    path.write_text(json.dumps({"events": [
        {"tStartMs": 1000, "dDurationMs": 2000,
         "segs": [{"utf8": "Hello &amp; welcome."}]},
    ]}), encoding="utf-8")
    captions = parse_json3_caption(path)
    assert [(s.text, s.start, s.end) for s in captions] == [
        ("Hello & welcome.", 1.0, 3.0)
    ]

    transcript = _to_transcript({"segments": [{
        "text": "Hello there. General Kenobi.", "start": 0.0, "end": 2.0,
        "speaker": "SPEAKER_00",
        "words": [
            {"word": "Hello", "start": 0.0, "end": 0.4, "speaker": "SPEAKER_00"},
            {"word": "there.", "start": 0.4, "end": 0.9, "speaker": "SPEAKER_00"},
            {"word": "General", "start": 1.0, "end": 1.4, "speaker": "SPEAKER_01"},
            {"word": "Kenobi.", "start": 1.4, "end": 2.0, "speaker": "SPEAKER_01"},
        ],
    }]}, language="en")
    assert [(s.speaker, s.text) for s in transcript.segments] == [
        ("SPEAKER_00", "Hello there."),
        ("SPEAKER_01", "General Kenobi."),
    ]


def test_manual_caption_path_skips_asr_and_audio(monkeypatch, tmp_path):
    import transcript

    caption = tmp_path / "video.en.json3"
    caption.write_text(json.dumps({"events": [
        {"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "Authored text."}]}
    ]}), encoding="utf-8")
    info = {"id": "video", "webpage_url": "https://youtube.com/watch?v=video"}

    monkeypatch.setattr(transcript, "download_manual_caption",
                        lambda *a, **k: (caption, "en", None, info))
    monkeypatch.setattr(transcript, "resolve_source",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("ASR download ran")))
    monkeypatch.setattr(transcript, "extract_audio",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("audio ran")))
    monkeypatch.setattr(transcript, "_ffmpeg_version", lambda: "6.0")

    class Engine:
        model_name = "large-v3"
        device = "cpu"
        compute_type = "int8"

        def run_captions(self, audio_path, captions, **kwargs):
            assert audio_path is None
            return Transcript(segments=captions, language=kwargs["language"])

    result = transcript.transcribe(
        "https://youtube.com/watch?v=video", diarize=False, engine=Engine(),
        work_dir=str(tmp_path),
    )
    assert result.text == "Authored text."
    assert result.meta["transcript_source"] == "youtube_manual_captions"
