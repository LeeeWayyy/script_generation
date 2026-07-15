from transcript import cli
from transcript.types import Segment, Transcript


def test_batch_reuses_one_engine_and_writes_indexed_outputs(monkeypatch, tmp_path):
    engine = object()
    calls = []
    monkeypatch.setattr(cli, "TranscriptionEngine", lambda **kwargs: engine)

    def fake_transcribe(source, **kwargs):
        calls.append((source, kwargs["engine"]))
        return Transcript(segments=[Segment(text=source)])

    monkeypatch.setattr(cli, "transcribe", fake_transcribe)

    assert cli.main(["first.mp3", "second.mp3", "--out-dir", str(tmp_path)]) == 0
    assert calls == [("first.mp3", engine), ("second.mp3", engine)]
    assert (tmp_path / "001-first.txt").read_text() == "first.mp3\n"
    assert (tmp_path / "002-second.txt").read_text() == "second.mp3\n"


def test_single_source_keeps_stdout_and_does_not_build_engine(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "TranscriptionEngine", lambda **kwargs: (_ for _ in ()).throw(AssertionError())
    )
    monkeypatch.setattr(
        cli, "transcribe", lambda source, **kwargs: Transcript(segments=[Segment(text="ok")])
    )

    assert cli.main(["one.mp3", "--no-diarize"]) == 0
    assert capsys.readouterr().out == "ok\n"


def test_output_write_failure_is_clean(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        cli, "transcribe", lambda source, **kwargs: Transcript(segments=[Segment(text="ok")])
    )

    output = tmp_path / "missing" / "out.txt"
    assert cli.main(["one.mp3", "--no-diarize", "-o", str(output)]) == 1
    assert f"Error: could not write {output}:" in capsys.readouterr().err


def test_stdout_write_failure_is_clean(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "transcribe", lambda source, **kwargs: Transcript(segments=[Segment(text="ok")])
    )
    monkeypatch.setattr(
        cli.sys.stdout, "write", lambda _text: (_ for _ in ()).throw(OSError("closed")),
    )

    assert cli.main(["one.mp3", "--no-diarize"]) == 1
    assert "Error: could not write output: closed" in capsys.readouterr().err
