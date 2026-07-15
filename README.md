# transcript

Generate transcripts — optionally with **speaker labels** — from any video or
audio source: a local file or a URL (YouTube and
[thousands of other sites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)
via yt-dlp).

Fully **self-hosted**. Speech recognition, word-level alignment, and speaker
diarization all run locally on your machine — nothing is uploaded.

> 📖 Full documentation: [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md) —
> architecture, API reference, deployment, security, and troubleshooting.

## Pipeline

```
YouTube ──► human captions ──► [align + diarize when enabled] ──► format
other/no captions ──► audio ──► WhisperX ──► align ──► diarize ──► format
```

YouTube auto-generated captions are ignored. When human captions exist, their
text is preserved and ASR is skipped; without diarization, audio is skipped too.

| Stage      | Tool                                   |
|------------|----------------------------------------|
| Download   | `yt-dlp`                               |
| Captions   | YouTube creator-supplied subtitles only |
| Audio      | `ffmpeg`                               |
| Transcribe | [WhisperX](https://github.com/m-bain/whisperX) (`large-v3`) |
| Speakers   | `pyannote` (bundled via WhisperX)      |

## Install

### 1. System tools

```bash
# macOS
brew install ffmpeg
# Debian/Ubuntu
sudo apt install ffmpeg
```

### 2. PyTorch — pick the build that matches your hardware

PyTorch is **not** pinned in `pyproject.toml` because the correct wheel depends
on your platform. Install it first, then this package.

```bash
# NVIDIA GPU (e.g. RTX 5090). Blackwell (sm_120) needs CUDA 12.8+:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# macOS / CPU-only
pip install torch torchaudio
```

> **RTX 5090 note:** Blackwell is very new. If you hit a
> `no kernel image is available` error, you need a newer PyTorch/CUDA build
> (CUDA 12.8+ from the [official PyTorch install matrix](https://pytorch.org/get-started/locally/)).
> Everything else stays the same.

### 3. This package

Pick the extra that matches how you'll use it — the base install pulls in no
heavy dependencies:

```bash
pip install -e ".[local]"    # local processing: `transcript` CLI + library
pip install -e ".[server]"   # GPU host: HTTP API (`transcript-server`)
pip install -e ".[client]"   # thin client only (`transcript-remote`, needs just requests)
# with dev tools:
pip install -e ".[local,dev]"
```

### 4. Hugging Face token (only for speaker labels)

Diarization uses gated pyannote models. One-time setup:

1. Create a token at <https://huggingface.co/settings/tokens>.
2. Accept the license for the diarization models. Which one you need depends on
   your WhisperX version:
   - WhisperX 3.8+ uses
     <https://huggingface.co/pyannote/speaker-diarization-community-1>.
   - Older WhisperX uses
     <https://huggingface.co/pyannote/speaker-diarization-3.1> (and the
     `segmentation-3.0` model it links to).

   Accepting access is usually instant. If you're unsure, accept both.
3. Expose it: `export HF_TOKEN=hf_xxx` (or pass `--hf-token` / `hf_token=`).

Music tagging is deliberately opt-in because it changes output. Install the
optional detector (`pip install -e ".[local,music]"`) and pass `--detect-music`.

## CLI usage

```bash
# Local file -> plain text with speaker labels, printed to stdout
transcript meeting.mp4

# YouTube URL -> SRT subtitles written to a file
transcript "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -f srt -o out.srt

# No speaker labels (skips diarization, no HF token needed), force English
transcript talk.mp3 --no-diarize --language en -f txt

# Structured JSON with word-level timestamps + speakers
transcript interview.wav -f json -o interview.json -v

# Give the diarizer hints when you know the speaker count
transcript panel.mp4 --min-speakers 2 --max-speakers 4

# Batch files with one warm model; deterministic collision-safe names in the directory
transcript a.mp4 b.mp4 --out-dir ./transcripts -f srt
```

Key flags: `-f/--format {txt,srt,vtt,json}`, `-o/--output`, `--out-dir`, `--model`,
`--no-diarize`, `--language`, `--device {cuda,cpu}`, `--min-speakers`, `--max-speakers`,
`--detect-music`, `-v/--verbose`.

## Library usage

```python
from transcript import transcribe

t = transcribe("meeting.mp4", diarize=True)

print(t.text)          # plain transcript
print(t.speakers)      # ['SPEAKER_00', 'SPEAKER_01']
print(t.language)      # 'en'

for seg in t.segments:
    print(f"[{seg.start:.1f}s] {seg.speaker}: {seg.text}")

# render to any format
from transcript.formats import render
open("out.srt", "w").write(render(t, "srt"))
```

Reuse a loaded model across many files (avoids reloading weights each call):

```python
from transcript import transcribe, TranscriptionEngine

engine = TranscriptionEngine(model="large-v3")  # auto-detects device
for f in ["a.mp4", "b.mp4", "c.mp4"]:
    print(transcribe(f, engine=engine).text)
```

## Remote: GPU server + thin client

Run the heavy work on your GPU box (e.g. a Windows PC with an RTX 5090) and call
it from a laptop (e.g. a Mac) on the same LAN. The client extra needs only
`requests` — no torch or WhisperX.

```
Mac (client)  ──HTTP/LAN──►  5090 PC (server + GPU)
 transcript-remote file.mp4      POST /jobs → queue → warm model runs → result
```

**On the GPU host** (Windows: see [`deploy/WINDOWS.md`](deploy/WINDOWS.md); Linux:
`deploy/run-server.sh` or the `deploy/transcript-server.service` systemd unit):

```bash
pip install -e ".[server]"
export HF_TOKEN=hf_xxx                  # for speaker labels
./deploy/run-server.sh                   # generates/persists auth token once
```

**On the Mac** (client):

```bash
pip install -e ".[client]"
export TRANSCRIPT_SERVER=http://<gpu-lan-ip>:8000
export TRANSCRIPT_TOKEN=<token printed by the server>

transcript-remote meeting.mp4 -f srt -o meeting.srt   # uploads the file
transcript-remote "https://youtube.com/watch?v=..." -f txt   # host downloads it

# extraction envelopes + verified assets
extract-remote slides.zip --kind image_note --out-dir ./extractions
extract-remote talk.mp4 --kind video --frames --cadence 10 --out-dir ./extractions
extract-remote --kind audio_extraction --feed-url https://example.com/feed.xml \
  --episode-guid episode-42 --out-dir ./extractions
```

`transcript-remote` uses the legacy transcript routes. `extract-remote` uses the
separate `/extractions` routes, downloads one zip, verifies its exact member set,
sizes, and SHA-256 hashes, then atomically publishes `<out-dir>/<job-id>`.
Completed extraction bundles live for seven days since last access under
`$TRANSCRIPT_DATA_DIR` (default `~/.cache/transcript/extractions`). See the full
documentation for the API, OCR model setup, cancellation, and cleanup controls.

Both clients accept `--detect-music`; it is off by default. When requested,
metadata records whether detection succeeded, detector version, the `0.5`
overlap threshold, and the number of segments flagged.

> Security: this is built for a trusted LAN with a shared token. Don't expose the
> port to the public internet — put it behind Tailscale/WireGuard or a TLS reverse
> proxy if you need off-LAN access.

## Cross-platform notes

`transcript` auto-detects the device: **CUDA if available, otherwise CPU**.

WhisperX's ASR runs through CTranslate2, which has **no Apple MPS backend** — so
on Apple Silicon it uses the CPU (this works, just slower than a CUDA GPU). Your
RTX 5090 is the fast path. Compute type defaults to `float16` on CUDA and `int8`
on CPU.

## Development

```bash
pip install -e ".[dev]"
pytest          # formatter + device tests run without the ML stack
ruff check .
```
