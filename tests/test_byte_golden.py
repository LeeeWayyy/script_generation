"""Byte-golden regression for the ASR meta-stamp (plan §0).

This is the REAL acceptance gate — not ``test_formats.py`` (which builds
``Transcript`` directly and never runs ``transcribe()``/``Worker.run``). It
drives the production functions through the public ``/jobs/{id}/result?format=json``
route with the ML/network sites monkeypatched, and pins the EXACT bytes (incl.
``meta`` key order) for both the URL and local-file shapes (they differ — the
local-file shape skips the download recipe).

The golden literals below were captured from the production path and verified to
follow the documented four-site insertion order:

  engine.run's meta.update  →  transcribe()'s first meta.update  →
  transcribe()'s URL-recipe meta.update (URL only)  →  Worker.run's meta.update

If a future edit reorders, adds, or drops a meta key — or leaks an extraction
field onto ``Transcript.meta`` — these byte comparisons break. That is the point.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _golden_harness import run_job_get_json  # noqa: E402

INFO = {
    "id": "GOLDEN",
    "webpage_url": "https://example.com/watch?v=GOLDEN",
    "format_id": "140",
    "channel": "Test Channel",
    "uploader": "Tester",
    "upload_date": "20240101",
}

GOLDEN_URL = """{
  "segments": [
    {
      "text": "Hello there.",
      "start": 0.0,
      "end": 1.5,
      "speaker": null,
      "music": false,
      "words": [
        {
          "word": "Hello",
          "start": 0.0,
          "end": 0.5,
          "score": 0.9,
          "speaker": null
        }
      ]
    },
    {
      "text": "General Kenobi.",
      "start": 1.6,
      "end": 3.2,
      "speaker": null,
      "music": false,
      "words": []
    }
  ],
  "language": "en",
  "meta": {
    "align_requested": true,
    "align_succeeded": true,
    "diarize_requested": false,
    "diarize_succeeded": null,
    "duration_s": 3.2,
    "whisperx_version": null,
    "pyannote_version": null,
    "source": "https://example.com/watch?v=GOLDEN",
    "model": "large-v3",
    "device": "cpu",
    "compute_type": "int8",
    "diarized": false,
    "video_id": "GOLDEN",
    "downloader": "yt-dlp",
    "yt_dlp_version": "2024.12.13",
    "ffmpeg_version": "6.0",
    "resolved_url": "https://example.com/watch?v=GOLDEN",
    "selected_format": "140",
    "channel": "Test Channel",
    "uploader": "Tester",
    "upload_date": "20240101",
    "job_id": "0123456789ab",
    "server_version": "0.1.0"
  }
}"""

GOLDEN_LOCAL = """{
  "segments": [
    {
      "text": "Hello there.",
      "start": 0.0,
      "end": 1.5,
      "speaker": null,
      "music": false,
      "words": [
        {
          "word": "Hello",
          "start": 0.0,
          "end": 0.5,
          "score": 0.9,
          "speaker": null
        }
      ]
    },
    {
      "text": "General Kenobi.",
      "start": 1.6,
      "end": 3.2,
      "speaker": null,
      "music": false,
      "words": []
    }
  ],
  "language": "en",
  "meta": {
    "align_requested": true,
    "align_succeeded": true,
    "diarize_requested": false,
    "diarize_succeeded": null,
    "duration_s": 3.2,
    "whisperx_version": null,
    "pyannote_version": null,
    "source": "/data/fixed-local-file.mp4",
    "model": "large-v3",
    "device": "cpu",
    "compute_type": "int8",
    "diarized": false,
    "job_id": "0123456789ab",
    "server_version": "0.1.0"
  }
}"""


def test_url_shape_byte_golden(monkeypatch):
    out = run_job_get_json(
        monkeypatch, source="https://example.com/watch?v=GOLDEN", info_json=INFO
    )
    assert out == GOLDEN_URL


def test_local_file_shape_byte_golden_skips_download_recipe(monkeypatch):
    out = run_job_get_json(
        monkeypatch, source="/data/fixed-local-file.mp4", info_json=None
    )
    assert out == GOLDEN_LOCAL
    # The download-recipe keys must be absent on the local-file shape.
    for k in ("video_id", "downloader", "yt_dlp_version", "resolved_url", "selected_format"):
        assert f'"{k}"' not in out
