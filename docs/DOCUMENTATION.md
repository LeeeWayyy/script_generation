# transcript — Full Documentation

Self-hosted transcript generation (with speaker labels) from any video or audio
source — a local file or a URL (YouTube and thousands of other sites). All
speech recognition, word alignment, and speaker diarization run locally; nothing
is uploaded to a third party.

- **Author:** leeewayyy
- **Version:** 0.1.0
- **License:** MIT

---

## Table of contents

1. [Overview](#1-overview)
2. [How it works (pipeline)](#2-how-it-works-pipeline)
3. [Architecture](#3-architecture)
4. [Installation](#4-installation)
5. [Usage — local CLI](#5-usage--local-cli)
6. [Usage — Python library](#6-usage--python-library)
7. [Usage — remote GPU server + client](#7-usage--remote-gpu-server--client)
8. [Output formats](#8-output-formats)
9. [Configuration reference](#9-configuration-reference)
10. [Deployment](#10-deployment)
11. [Performance & hardware](#11-performance--hardware)
12. [Security](#12-security)
13. [Troubleshooting](#13-troubleshooting)
14. [Project layout](#14-project-layout)
15. [Development](#15-development)
16. [Roadmap](#16-roadmap)

---

## 1. Overview

`transcript` converts speech in any media into text. It is designed around three
requirements:

- **Any source** — local files (mp4, mov, mp3, wav, m4a, …) or URLs (downloaded
  with `yt-dlp`).
- **Accurate, timestamped text** — creator-supplied YouTube captions when
  available, otherwise [WhisperX](https://github.com/m-bain/whisperX)
  (`large-v3` by default), including word-level timestamps after alignment.
- **Speaker labels** — "who said what", via `pyannote` diarization (bundled with
  WhisperX).

It ships in three usable shapes from one codebase:

| Shape | Entry point | Who uses it |
|-------|-------------|-------------|
| Python library | `from transcript import transcribe` | other code / notebooks |
| Local CLI | `transcript <source>` | run on the machine with the GPU |
| Remote client/server | `transcript-server` + `transcript-remote` | GPU host + laptop on a LAN |

---

## 2. How it works (pipeline)

YouTube uses creator-supplied captions when available; auto-generated captions
are ignored. Other sources, and YouTube videos without matching human captions,
use ASR:

```
YouTube ──► human captions ──► [align + diarize when enabled] ──► format
other/no captions ──► audio ──► WhisperX ──► align ──► diarize ──► format
```

| Stage | Module | Tool | Notes |
|-------|--------|------|-------|
| Ingest | `ingest.py` | `yt-dlp` | Uses matching human YouTube captions; otherwise downloads best audio. |
| Audio  | `audio.py` | `ffmpeg` | Normalizes anything to 16 kHz mono PCM WAV. |
| Transcribe | `engine.py` | WhisperX (CTranslate2) | Batched Whisper inference with timestamps. |
| Align | `engine.py` | WhisperX align model | Word-level timing; per-language model, cached. |
| Diarize | `engine.py` | `pyannote` | Detects speakers; assigns each word a speaker. |
| Format | `formats.py` | — | Renders txt / srt / vtt / json. |

The result is a `Transcript` dataclass (`types.py`): a list of `Segment`s, each
with text, start/end, an optional `speaker`, and a list of `Word`s.

---

## 3. Architecture

### 3.1 Module map

```
src/transcript/
  __init__.py    Public API: transcribe(...) — orchestrates the whole pipeline.
  types.py       Transcript / Segment / Word dataclasses (dependency-free).
  device.py      CUDA/CPU auto-detection + compute-type selection.
  ingest.py      Source -> local media file (local path or yt-dlp download).
  audio.py       Media file -> 16 kHz mono WAV via ffmpeg.
  engine.py      TranscriptionEngine: WhisperX transcribe -> align -> diarize.
  formats.py     Transcript -> txt / srt / vtt / json.
  cli.py         `transcript` command (runs locally, on the GPU machine).
  server.py      `transcript-server` HTTP API (runs on the GPU host).
  remote.py      `transcript-remote` thin client (runs on the laptop).
```

### 3.2 Design decisions

- **Lazy heavy imports.** `torch` / `whisperx` are imported *inside* functions in
  `engine.py`, never at module top level. So importing the package, running
  `transcript --help`, or running the test suite does **not** load the multi-GB
  ML stack. The client (`remote.py`) never imports them at all.
- **Library core, thin wrappers.** `cli.py`, `server.py`, and `remote.py` are all
  thin layers over the single `transcribe()` function. Behavior stays consistent
  across every entry point.
- **Engine reuse / warm model.** `TranscriptionEngine` loads model weights once
  and caches alignment models per language. Pass an existing engine to
  `transcribe(..., engine=eng)` to process many files without reloading. The
  server keeps one warm engine for its lifetime.
- **Auto-detection with an honest fallback.** WhisperX's ASR runs through
  CTranslate2, which supports **CPU and CUDA only — no Apple MPS**. `device.py`
  therefore never returns `"mps"`; on Apple Silicon it selects CPU and logs a
  warning. CUDA is the fast path.
- **`torch` left unpinned.** The correct wheel is platform/CUDA-specific, so it
  is installed manually (see §4), not declared as a hard dependency.

### 3.3 Remote topology

```
┌────────────── LAN ──────────────┐
│                                  │
│  Mac (client)        Windows PC / Linux box (server + GPU)
│  ┌──────────────┐    ┌──────────────────────────────────┐
│  │ transcript-  │    │ transcript-server (FastAPI)       │
│  │ remote       │    │  ├ POST /jobs   (upload | url)     │
│  │  (requests   │───►│  ├ GET  /jobs/{id}                 │
│  │   only)      │◄───│  ├ GET  /jobs/{id}/result?format= │
│  └──────────────┘    │  └ GET  /health                   │
│                      │     │                              │
│                      │     ▼ background worker (1 GPU job │
│                      │       at a time, warm model)       │
│                      └──────────────────────────────────┘
└──────────────────────────────────────────────────────────┘
```

Jobs are queued in memory and processed serially by a single background worker,
so the GPU is used one job at a time and the model stays resident. The client
submits, polls status, then fetches the rendered result in the format it wants.

---

## 4. Installation

### 4.1 System tools (all machines that run the pipeline)

`ffmpeg` is required; `yt-dlp` is installed as a Python dependency.

```bash
# macOS
brew install ffmpeg
# Debian / Ubuntu
sudo apt install ffmpeg
# Windows
winget install Gyan.FFmpeg
```

### 4.2 PyTorch — install the wheel that matches your hardware

PyTorch is **not** pinned by the project because the right build is
platform-specific. Install it first.

```bash
# NVIDIA GPU (e.g. RTX 5090 — Blackwell/sm_120 needs a recent CUDA wheel)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# macOS / CPU-only
pip install torch torchaudio
```

### 4.3 The package — pick the extras you need

```bash
# Full local install (run the pipeline on this machine):
pip install -e .

# GPU host that will serve the API:
pip install -e ".[server]"

# Laptop that only talks to a remote server (no torch/whisperx needed):
pip install requests          # or:  pip install -e ".[client]"

# Contributors:
pip install -e ".[dev]"
```

### 4.4 Hugging Face token (only for speaker labels)

Diarization uses gated `pyannote` models:

1. Create a token at <https://huggingface.co/settings/tokens>.
2. Accept the licenses at
   <https://huggingface.co/pyannote/speaker-diarization-3.1> and the
   `segmentation-3.0` model it links to.
3. Expose it: `export HF_TOKEN=hf_xxx` (Windows: `setx HF_TOKEN "hf_xxx"`), or
   pass `--hf-token` / `hf_token=`.

Without a token you can still transcribe — just run with `--no-diarize`.

---

## 5. Usage — local CLI

Run on the machine that has the GPU/ffmpeg installed.

```bash
# Local file -> plain text with speaker labels, to stdout
transcript meeting.mp4

# YouTube URL -> SRT subtitles to a file
transcript "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -f srt -o out.srt

# No diarization (no HF token needed), force English
transcript talk.mp3 --no-diarize --language en

# Structured JSON with word timestamps + speakers, verbose progress
transcript interview.wav -f json -o interview.json -v

# Hint the speaker count for better diarization
transcript panel.mp4 --min-speakers 2 --max-speakers 4
```

See §9 for every flag.

---

## 6. Usage — Python library

```python
from transcript import transcribe

t = transcribe("meeting.mp4", diarize=True)

print(t.text)          # plain transcript (no timestamps)
print(t.speakers)      # ['SPEAKER_00', 'SPEAKER_01']
print(t.language)      # 'en'

for seg in t.segments:
    print(f"[{seg.start:.1f}s] {seg.speaker}: {seg.text}")
```

Render to a file in any format:

```python
from transcript.formats import render
open("out.srt", "w").write(render(t, "srt"))
```

Reuse a warm model across many files:

```python
from transcript import transcribe, TranscriptionEngine

engine = TranscriptionEngine(model="large-v3")   # auto-detects device
for f in ["a.mp4", "b.mp4", "c.mp4"]:
    print(transcribe(f, engine=engine).text)
```

---

## 7. Usage — remote GPU server + client

Run the heavy work on the GPU box; drive it from a laptop on the same LAN.

### 7.1 On the GPU host

```bash
pip install -e ".[server]"
export HF_TOKEN=hf_xxx
export TRANSCRIPT_TOKEN=$(python -c "import secrets;print(secrets.token_urlsafe(24))")
transcript-server --host 0.0.0.0 --port 8000
```

Windows users: use `deploy\run-server.ps1` (generates the token, opens the
firewall, prints the LAN URL). Full walkthrough in
[`deploy/WINDOWS.md`](../deploy/WINDOWS.md).

### 7.2 On the laptop

```bash
pip install requests
export TRANSCRIPT_SERVER=http://<gpu-lan-ip>:8000
export TRANSCRIPT_TOKEN=<token the server printed>

curl $TRANSCRIPT_SERVER/health                              # confirm reachable
transcript-remote meeting.mp4 -f srt -o meeting.srt         # uploads the file
transcript-remote "https://youtube.com/watch?v=..." -f txt  # host downloads it
```

For a URL job, nothing uploads from the laptop — the GPU host fetches the media
directly, which is faster.

### 7.3 HTTP API reference

| Method & path | Auth | Body / params | Returns |
|---------------|------|---------------|---------|
| `GET /health` | none | — | `{status, model, queued_or_running[]}` |
| `POST /jobs` | bearer | multipart: `file` **or** `url`; optional `diarize`, `language`, `min_speakers`, `max_speakers` | job JSON incl. `id`, `status` |
| `GET /jobs` | bearer | — | list of jobs |
| `GET /jobs/{id}` | bearer | — | job JSON (`status`: queued/running/done/error) |
| `GET /jobs/{id}/result` | bearer | `format=txt\|srt\|vtt\|json` | rendered transcript (text/plain) |

Auth: when `TRANSCRIPT_TOKEN` is set, every endpoint except `/health` requires
`Authorization: Bearer <token>`. If it is unset the server runs **open** and logs
a warning — only acceptable on a trusted, firewalled LAN.

Example with curl:

```bash
# submit a file
curl -H "Authorization: Bearer $TRANSCRIPT_TOKEN" \
     -F file=@meeting.mp4 -F diarize=true \
     $TRANSCRIPT_SERVER/jobs
# -> {"id":"ab12cd34ef56", "status":"queued", ...}

# poll
curl -H "Authorization: Bearer $TRANSCRIPT_TOKEN" \
     $TRANSCRIPT_SERVER/jobs/ab12cd34ef56

# fetch result as SRT
curl -H "Authorization: Bearer $TRANSCRIPT_TOKEN" \
     "$TRANSCRIPT_SERVER/jobs/ab12cd34ef56/result?format=srt"
```

---

## 8. Output formats

| Format | Extension | Contents |
|--------|-----------|----------|
| `txt`  | `.txt` | Plain transcript, one line per segment, `SPEAKER_xx:` prefix when diarized. |
| `srt`  | `.srt` | SubRip subtitles, numbered cues, `HH:MM:SS,mmm` timestamps, `[SPEAKER_xx]` prefix. |
| `vtt`  | `.vtt` | WebVTT subtitles, `HH:MM:SS.mmm` timestamps. |
| `json` | `.json` | Full structured data: segments, words, timestamps, scores, speakers, language, meta. |

The JSON form is the canonical, lossless output; the others are derived views.

---

## 9. Configuration reference

### 9.1 CLI flags (`transcript`)

| Flag | Default | Meaning |
|------|---------|---------|
| `source` (positional) | — | Local file path or http(s) URL. |
| `-f, --format` | `txt` | `txt` / `srt` / `vtt` / `json`. |
| `-o, --output` | stdout | Write to this file. |
| `--model` | `large-v3` | Whisper model name (e.g. `medium`, `small`). |
| `--diarize` / `--no-diarize` | on | Speaker labels on/off. |
| `--language` | auto | Force a language code (e.g. `en`). |
| `--device` | auto | `cuda` or `cpu`. |
| `--compute-type` | auto | CTranslate2 compute type (e.g. `float16`, `int8`). |
| `--hf-token` | `$HF_TOKEN` | Hugging Face token for diarization. |
| `--min-speakers` / `--max-speakers` | none | Diarization hints. |
| `--batch-size` | 16 | ASR batch size. |
| `--no-align` | off | Skip word alignment (faster, coarser timing). |
| `-v, --verbose` | off | Verbose logging / full tracebacks. |

### 9.2 `transcribe()` parameters

Mirror the CLI flags, plus: `work_dir` (where downloads/temp audio live),
`keep_audio` (don't delete temp files), and `engine` (reuse a warm
`TranscriptionEngine`).

### 9.3 Environment variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `HF_TOKEN` / `HUGGINGFACE_TOKEN` | server, engine | Diarization model auth. |
| `TRANSCRIPT_TOKEN` | server, client | Bearer auth token. |
| `TRANSCRIPT_SERVER` | client | Server base URL. |
| `TRANSCRIPT_HOST` / `TRANSCRIPT_PORT` / `TRANSCRIPT_MODEL` | launch scripts | Server bind + model. |

---

## 10. Deployment

| Target | Asset | Notes |
|--------|-------|-------|
| Windows (5090 PC) | `deploy/run-server.ps1`, `deploy/run-server.bat`, `deploy/WINDOWS.md` | Token gen, firewall, LAN-IP hint; NSSM / Task Scheduler for auto-start. |
| Linux | `deploy/run-server.sh` | Token gen + launch. |
| Linux (service) | `deploy/transcript-server.service` | systemd unit; auto-start on boot. |

To keep the server running unattended:
- **Windows:** NSSM service or a Task Scheduler "at startup" task (see
  `deploy/WINDOWS.md`).
- **Linux:** the provided systemd unit (`systemctl enable --now transcript-server`).

---

## 11. Performance & hardware

- **NVIDIA GPU (CUDA)** is the intended runtime. With `large-v3` + `float16`,
  transcription runs many times faster than realtime; diarization adds overhead
  but stays well below realtime on a modern GPU.
- **RTX 5090 (Blackwell, sm_120)** is very new — use a CUDA 12.4+ (or nightly)
  PyTorch wheel, or you may see `no kernel image is available`.
- **Apple Silicon / CPU** works but uses CPU for ASR (no MPS backend in
  CTranslate2) and CPU for diarization. Fine for short clips; slower for long
  media. Compute type defaults to `int8` on CPU.
- **Throughput knobs:** larger `--batch-size` (more VRAM, faster); a smaller
  `--model` (faster, less accurate); `--no-align` and `--no-diarize` skip stages.

---

## 12. Security

- The remote API is built for a **trusted LAN with a shared bearer token over
  plain HTTP**. That is appropriate inside a home/office network.
- **Do not** expose the port directly to the public internet. For off-LAN access,
  put it behind **Tailscale/WireGuard** (simplest) or a **TLS reverse proxy**
  that enforces the token.
- The launch scripts open the firewall only for the **Private** network profile.
- Uploaded files are written to a temp directory and deleted after the job
  completes. Downloaded URL media lives in a temp work dir cleaned up per call.
- Keep `HF_TOKEN` and `TRANSCRIPT_TOKEN` out of source control (they belong in
  environment variables / service config).

---

## 13. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Required tool 'ffmpeg' was not found` | ffmpeg not installed / not on PATH | Install ffmpeg (§4.1); reopen the terminal. |
| `Speaker diarization needs a Hugging Face token` | `HF_TOKEN` unset or license not accepted | Set the token and accept the pyannote licenses (§4.4), or use `--no-diarize`. |
| `no kernel image is available for execution` | torch wheel too old for your GPU | Install a cu124+/nightly wheel (§4.2). |
| `compute_type float16 ... unsupported` on CPU | float16 on a CPU build | Handled automatically (retries int8); or pass `--compute-type int8`. |
| Mac can't reach the server | firewall / network profile / wrong IP | Set LAN to Private, run PowerShell as admin once, `curl /health` with the printed IP. |
| `401 unauthorized` from the server | missing/incorrect token | Set `--token` / `$TRANSCRIPT_TOKEN` to match the server. |
| yt-dlp download fails | site change / no network / age-gate | Update `yt-dlp` (`pip install -U yt-dlp`); check the URL. |
| Word timestamps missing | alignment failed for that language | Non-fatal; transcript still produced. Check `-v` logs. |

---

## 14. Project layout

```
transcript/
├── pyproject.toml                  Packaging; CLI/server/client entry points; extras.
├── README.md                       Quick start.
├── docs/
│   └── DOCUMENTATION.md            This document.
├── deploy/
│   ├── run-server.ps1              Windows launcher (token, firewall, LAN IP).
│   ├── run-server.bat             Windows launcher (minimal).
│   ├── run-server.sh               Linux launcher.
│   ├── transcript-server.service   systemd unit (Linux).
│   └── WINDOWS.md                  Windows host setup guide.
├── src/transcript/
│   ├── __init__.py                 Public API: transcribe(...).
│   ├── types.py                    Transcript / Segment / Word.
│   ├── device.py                   Device + compute-type detection.
│   ├── ingest.py                   Source -> media file.
│   ├── audio.py                    Media -> 16 kHz mono WAV.
│   ├── engine.py                   WhisperX transcribe/align/diarize.
│   ├── formats.py                  Renderers.
│   ├── cli.py                      `transcript` command.
│   ├── server.py                   `transcript-server` API.
│   └── remote.py                   `transcript-remote` client.
└── tests/
    ├── test_formats.py             Formatter + data-model tests.
    └── test_device.py              Device selection tests.
```

---

## 15. Development

```bash
pip install -e ".[dev]"
pytest          # formatter + device tests; run without the ML stack
ruff check .
```

The test suite intentionally covers the dependency-free parts (formatters, data
model, device logic) so it runs fast anywhere, including CI without a GPU.

Conventions:
- Keep `torch`/`whisperx` imports lazy (inside functions in `engine.py`).
- New entry points should wrap `transcribe()` rather than reimplement the
  pipeline.
- Line length 100; `ruff` configured in `pyproject.toml`.

---

## 16. Roadmap

Ideas not yet implemented:

- Batch mode (`transcript *.mp4 --out-dir ./transcripts/`) with one model load.
- A browser upload page served by the same server (drag-and-drop, no CLI).
- Job persistence (survive a server restart) and result caching by content hash.
- Optional model warmup on startup so the first job isn't slow.
- An end-to-end smoke test that runs a short clip through the whole pipeline.
- Output enrichment: paragraph segmentation, speaker renaming, simple summaries.
```
