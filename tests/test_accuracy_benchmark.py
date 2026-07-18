import hashlib
import json

import pytest

from benchmarks.run import (
    edit_distance,
    gate_failures,
    load_cases,
    normalize_text,
    score,
    summarize,
)


def test_accuracy_metrics_and_manifest(tmp_path):
    assert normalize_text("Don’t STOP—ＦＯＯ １２!") == "dont stop foo 12"
    assert edit_distance(["one", "two"], ["one", "three", "two"]) == 1
    metrics = score("one two three", "one too three four")
    assert metrics["word_errors"] == 2
    assert metrics["wer"] == pytest.approx(2 / 3)

    media = tmp_path / "sample.wav"
    media.write_bytes(b"audio")
    reference = tmp_path / "sample.txt"
    reference.write_text("one two three", encoding="utf-8")
    manifest = tmp_path / "cases.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "cases": [{
            "id": "sample",
            "media": media.name,
            "media_sha256": hashlib.sha256(b"audio").hexdigest(),
            "reference": reference.name,
            "duration_s": 3.0,
            "language": "en",
        }],
    }), encoding="utf-8")
    cases = load_cases(manifest)
    assert cases[0].reference == "one two three"

    row = {"run": 1, "elapsed_s": 1.5, **metrics}
    summary = summarize([row], cases, runs=1)
    assert summary["corpus_wer"] == pytest.approx(2 / 3)
    assert summary["realtime_factor"] == pytest.approx(0.5)
    assert gate_failures(summary, max_wer=0.7, max_rtf=0.6) == []
    assert gate_failures(summary, max_wer=0.5, max_rtf=0.4) == [
        "WER 66.67% exceeds 50.00%", "RTF 0.500 exceeds 0.400",
    ]

    bad = json.loads(manifest.read_text(encoding="utf-8"))
    bad["cases"][0]["media_sha256"] = "0" * 64
    manifest.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="media_sha256"):
        load_cases(manifest)
