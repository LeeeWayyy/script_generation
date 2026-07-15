"""Opt-in system/ML smoke test; skipped in the fast suite and CI by default."""

import os
import shutil
import subprocess

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("TRANSCRIPT_REAL_MEDIA_SMOKE") != "1",
    reason="set TRANSCRIPT_REAL_MEDIA_SMOKE=1 to run the real media pipeline",
)


def test_real_media_pipeline(tmp_path):
    """Exercise ffmpeg + WhisperX, and yt-dlp too when TRANSCRIPT_SMOKE_URL is set."""
    from transcript import transcribe

    source = os.environ.get("TRANSCRIPT_SMOKE_URL")
    if not source:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            pytest.skip("ffmpeg is not installed")
        source = str(tmp_path / "tone.wav")
        subprocess.run(
            [
                ffmpeg, "-loglevel", "error", "-f", "lavfi", "-i",
                "sine=frequency=440:duration=1", "-ar", "16000", "-ac", "1", source,
            ],
            check=True,
        )

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    result = transcribe(
        source,
        model=os.environ.get("TRANSCRIPT_SMOKE_MODEL", "tiny"),
        diarize=False,
        work_dir=str(work_dir),
    )
    assert result.meta["source"] == source
