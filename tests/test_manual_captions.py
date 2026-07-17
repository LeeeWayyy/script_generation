import json
import subprocess

from transcript.engine import _to_transcript
from transcript.ingest import (
    _manual_caption_language,
    download_manual_caption,
    parse_json3_caption,
)
from transcript.types import Segment, Transcript


def test_auto_captions_are_ignored(monkeypatch, tmp_path):
    info = {"id": "video", "automatic_captions": {"en": [{"ext": "json3"}]}}
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        assert kwargs["monitor_stdout"] is True
        return json.dumps(info), 100

    monkeypatch.setattr("transcript.ingest._run_ytdlp", fake_run)

    assert download_manual_caption(
        "https://youtube.com/watch?v=video", tmp_path, language="en"
    ) is None
    assert len(calls) == 1


def test_manual_caption_probe_and_subtitle_share_bounded_runner(monkeypatch, tmp_path):
    info = {"id": "video", "subtitles": {"en": [{"ext": "json3"}]}}
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if len(calls) == 1:
            return json.dumps(info), 100
        (tmp_path / "video.en.json3").write_text("{}", encoding="utf-8")
        return "", 100

    monkeypatch.setattr("transcript.ingest._run_ytdlp", fake_run)
    result = download_manual_caption(
        "https://youtube.com/watch?v=video", tmp_path, language="en",
    )

    assert result is not None and result[0] == tmp_path / "video.en.json3"
    assert [kwargs["fail_action"] for _, kwargs in calls] == [
        "inspect", "download manual captions",
    ]
    assert calls[0][1]["monitor_stdout"] is True
    assert "--write-subs" in calls[1][0]


def test_manual_caption_partial_file_is_capped_and_cleaned(monkeypatch, tmp_path):
    info = {"id": "video", "subtitles": {"en": [{"ext": "json3"}]}}
    partial = tmp_path / "video.en.json3.part"
    calls = []

    class Process:
        returncode = None

        def __init__(self, cmd, **_kwargs):
            self.cmd = cmd
            self.killed = False
            calls.append(cmd)

        def communicate(self, timeout=None):
            if self.killed:
                self.returncode = -9
                return "", ""
            if "--dump-single-json" in self.cmd:
                self.returncode = 0
                return json.dumps(info), ""
            partial.write_bytes(b"x" * 101)
            raise subprocess.TimeoutExpired(self.cmd, timeout)

        def kill(self):
            self.killed = True

    monkeypatch.setenv("TRANSCRIPT_MAX_DOWNLOAD_BYTES", "100")
    monkeypatch.setattr("transcript.ingest.subprocess.Popen", Process)

    result = download_manual_caption(
        "https://youtube.com/watch?v=video", tmp_path, language="en",
    )

    assert result == (None, "en", None, info)
    assert not partial.exists()
    assert "--write-subs" in calls[1]
    assert all("--max-filesize" not in cmd for cmd in calls)


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


def test_manual_caption_alignment_skips_asr_and_uses_audio(monkeypatch, tmp_path):
    import transcript

    caption = tmp_path / "video.en.json3"
    caption.write_text(json.dumps({"events": [
        {"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "Authored text."}]}
    ]}), encoding="utf-8")
    media = tmp_path / "video.webm"
    audio = tmp_path / "video.wav"
    info = {"id": "video", "webpage_url": "https://youtube.com/watch?v=video"}

    def download(*args, **kwargs):
        assert kwargs["with_audio"] is True
        return caption, "en", media, info

    def extract(media_path, *_args):
        assert media_path == media
        return audio

    monkeypatch.setattr(transcript, "download_manual_caption", download)
    monkeypatch.setattr(transcript, "resolve_source",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("ASR download ran")))
    monkeypatch.setattr(transcript, "extract_audio", extract)
    monkeypatch.setattr(transcript, "_ffmpeg_version", lambda: "6.0")

    class Engine:
        model_name = "large-v3"
        device = "cpu"
        compute_type = "int8"

        def run_captions(self, audio_path, captions, **kwargs):
            assert audio_path == str(audio)
            assert kwargs["align"] is True
            return Transcript(segments=captions, language=kwargs["language"])

    result = transcript.transcribe(
        "https://youtube.com/watch?v=video", diarize=False, engine=Engine(),
        work_dir=str(tmp_path),
    )
    assert result.text == "Authored text."
    assert result.meta["transcript_source"] == "youtube_manual_captions"
    assert "music_detection_requested" not in result.meta


def test_manual_caption_music_opt_in_acquires_audio_and_stamps_provenance(
    monkeypatch, tmp_path,
):
    import transcript

    caption = tmp_path / "video.en.json3"
    caption.write_text(json.dumps({"events": [
        {"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "Song."}]}
    ]}), encoding="utf-8")
    media = tmp_path / "video.webm"
    audio = tmp_path / "video.wav"

    def download(*args, **kwargs):
        assert kwargs["with_audio"] is True
        return caption, "en", media, {"id": "video"}

    monkeypatch.setattr(transcript, "download_manual_caption", download)
    monkeypatch.setattr(transcript, "extract_audio", lambda *a, **k: audio)
    monkeypatch.setattr(transcript, "_ffmpeg_version", lambda: "6.0")
    monkeypatch.setattr("transcript.music.detect_and_tag", lambda result, path: 1)
    monkeypatch.setattr("transcript.music.detector_version", lambda: "0.8.0")

    class Engine:
        model_name = "large-v3"
        device = "cpu"
        compute_type = "int8"

        def run_captions(self, audio_path, captions, **kwargs):
            assert audio_path == str(audio)
            assert kwargs["diarize"] is False
            return Transcript(segments=captions, language="en")

    result = transcript.transcribe(
        "https://youtube.com/watch?v=video", diarize=False, detect_music=True,
        engine=Engine(), work_dir=str(tmp_path),
    )
    assert result.meta["music_detection_requested"] is True
    assert result.meta["music_detection_succeeded"] is True
    assert result.meta["music_detector_version"] == "0.8.0"
    assert result.meta["music_overlap_threshold"] == 0.5
    assert result.meta["music_segments_flagged"] == 1


def test_caption_without_audio_preserves_alignment_request_metadata():
    from transcript.engine import TranscriptionEngine

    engine = object.__new__(TranscriptionEngine)
    result = engine.run_captions(
        None, [Segment(text="caption", start=0.0, end=1.0)], language="en",
        diarize=False, align=True,
    )
    assert result.meta["align_requested"] is True
    assert result.meta["align_succeeded"] is None


def test_alignment_cache_retains_only_the_last_language(monkeypatch):
    import sys
    from types import SimpleNamespace

    from transcript.engine import TranscriptionEngine

    calls = []

    def load_align_model(*, language_code, device):
        calls.append(language_code)
        return (f"model-{language_code}", {"language": language_code})

    monkeypatch.setitem(sys.modules, "whisperx", SimpleNamespace(load_align_model=load_align_model))
    engine = object.__new__(TranscriptionEngine)
    engine.device = "cpu"
    engine._align_cache = None

    assert engine._load_align("en") == engine._load_align("en")
    engine._load_align("fr")
    engine._load_align("en")
    assert calls == ["en", "fr", "en"]
    assert engine._align_cache[0] == "en"
