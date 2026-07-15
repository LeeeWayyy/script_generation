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
14. [Development](#14-development)

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

It ships in four usable shapes from one codebase:

| Shape | Entry point | Who uses it |
|-------|-------------|-------------|
| Python library | `from transcript import transcribe` | other code / notebooks |
| Local CLI | `transcript <source>` | run on the machine with the GPU |
| Remote transcript | `transcript-server` + `transcript-remote` | rendered txt/srt/vtt/json |
| Remote extraction | `transcript-server` + `extract-remote` | provenance envelope + verified assets |

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

The separate extraction pipeline handles three explicit kinds:

- `video`: ASR plus optional sampled frames and OCR.
- `image_note`: ordered images from a zip/tar manual export plus OCR.
- `audio_extraction`: a podcast episode selected from RSS provenance, then ASR.

The result is a `Transcript` dataclass (`types.py`): a list of `Segment`s, each
with text, start/end, an optional `speaker`, and a list of `Word`s.
Extraction results instead use their own `ExtractionResult` JSON envelope and
never pass through the legacy transcript serializer.

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
  extraction.py  ExtractionResult envelope + canonical extraction renderer.
  extract.py     Video, image-note, and podcast extraction orchestration.
  extract_remote.py  `extract-remote` client + verified atomic bundle unpack.
  extraction_store.py  Durable extraction bundles + TTL cleanup.
  archive.py / frames.py / ocr.py / podcast.py  Modality-specific helpers.
```

### 3.2 Design decisions

- **Lazy heavy imports.** `torch` / `whisperx` are imported *inside* functions in
  `engine.py`, never at module top level. So importing the package, running
  `transcript --help`, or running the test suite does **not** load the multi-GB
  ML stack. The client (`remote.py`) never imports them at all.
- **Shared core, separate contracts.** Local/remote ASR uses `transcribe()` and
  the stable transcript renderers. Extraction routes use a separate envelope and
  durable store, while reusing the same ASR engine where appropriate.
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
│  │  (requests   │───►│  ├ /jobs/* transcript routes      │
│  │   only)      │◄───│  ├ /extractions/* bundle routes   │
│  └──────────────┘    │  └ GET /health                    │
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
# NVIDIA GPU (e.g. RTX 5090 — Blackwell/sm_120 needs CUDA 12.8+)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# macOS / CPU-only
pip install torch torchaudio
```

### 4.3 The package — pick the extras you need

```bash
# Full local install (run the pipeline on this machine):
pip install -e ".[local]"

# GPU host that will serve the API:
pip install -e ".[server]"

# Laptop that only talks to a remote server (no torch/whisperx needed):
pip install -e ".[client]"

# Contributors:
pip install -e ".[dev]"
```

The console scripts are installed by this project. `pip install requests` alone
does **not** install `transcript-remote` or `extract-remote`.

Music classification is an optional, potentially conflicting ML stack. Install
it only on a host that will use explicit music detection:

```bash
pip install -e ".[local,music]"   # local CLI
pip install -e ".[server,music]"  # server
```

### 4.4 Hugging Face token (only for speaker labels)

Diarization uses gated `pyannote` models:

1. Create a token at <https://huggingface.co/settings/tokens>.
2. Accept the license for the model used by your WhisperX version:
   [speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   for WhisperX 3.8+, or
   [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   plus its linked `segmentation-3.0` model for older versions. Accept both if
   you are unsure.
3. Expose it: `export HF_TOKEN=hf_xxx` (Windows: `setx HF_TOKEN "hf_xxx"`), or
   pass `--hf-token` / `hf_token=`.

Without a token you can still transcribe — just run with `--no-diarize`.

### 4.5 OCR model setup (server extraction only)

`image_note` and video `--frames` use PaddleOCR. The server deliberately refuses
an implicit model download during a job. Choose one deployment mode:

```bash
# Simple trusted setup: explicitly allow PaddleOCR's first-run download.
export TRANSCRIPT_OCR_ALLOW_DOWNLOAD=1

# Reproducible/offline setup: provide pre-fetched PaddleOCR 2.x Chinese-model
# directories named det/, rec/, and cls/ under one root.
export TRANSCRIPT_OCR_MODEL_DIR=/srv/transcript/ocr
find "$TRANSCRIPT_OCR_MODEL_DIR" -maxdepth 1 -type d
# .../det  .../rec  .../cls
```

The pinned directories must match the PaddleOCR 2.x versions in the `server`
extra. OCR initialization failure does not discard the extraction: image/frame
assets remain, OCR text is empty, and metadata reports `ocr_requested`,
`ocr_succeeded`, and a non-sensitive `ocr_warning`.

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

# Opt in to music tagging (requires the music extra; rendered views mark [MUSIC])
transcript concert.mp4 --detect-music -f vtt -o concert.vtt

# Batch sources reuse one warm model and write numbered, collision-safe names.
transcript a.mp4 b.mp4 --out-dir ./transcripts -f srt
```

Multiple sources require `--out-dir` and stop at the first failure. One
`TranscriptionEngine` is initialized and reused across the batch. Output names
are `001-<sanitized-source-stem>.<format>`, `002-...`, so duplicate stems do not
collide. A single source can also use `--out-dir`; `--out-dir` and `-o/--output`
are mutually exclusive.

See §9 for every flag.

---

## 6. Usage — Python library

```python
from transcript import transcribe

t = transcribe("meeting.mp4", diarize=True, detect_music=False)

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
./deploy/run-server.sh
```

The launcher generates an owner-only token file on first run and reuses it after
restarts. Windows users: use `deploy\run-server.ps1` (same persistence, plus the
firewall and LAN URL). Full walkthrough in
[`deploy/WINDOWS.md`](../deploy/WINDOWS.md).

### 7.2 On the laptop

```bash
pip install -e ".[client]"
export TRANSCRIPT_SERVER=http://<gpu-lan-ip>:8000
export TRANSCRIPT_TOKEN=<token the server printed>

curl $TRANSCRIPT_SERVER/health                              # confirm reachable
transcript-remote meeting.mp4 -f srt -o meeting.srt         # uploads the file
transcript-remote "https://youtube.com/watch?v=..." -f txt  # host downloads it
```

For a URL job, nothing uploads from the laptop — the GPU host fetches the media
directly. `--timeout` controls submission/upload, each connection/read, polling,
and result download; `--poll` controls the status interval. Progress is shown on
stderr unless `--quiet` is used. Speaker hints, `--language`, `--no-diarize`, and
`--detect-music` are sent to the server.

### 7.3 Extraction client (`extract-remote`)

The extraction client returns canonical candidate text on stdout and atomically
publishes the verified envelope/assets under `<out-dir>/<job-id>/`:
```bash
# Ordered images from a zip/tar manual export, with OCR.
extract-remote cards.zip --kind image_note --out-dir ./results

# Video ASR plus frames sampled every 10 seconds and OCR'd.
extract-remote video.mp4 --kind video --frames --cadence 10 \
  --min-speakers 2 --max-speakers 4 --out-dir ./results

# Podcast RSS provenance: select by GUID, episode URL, or title+published pair.
extract-remote --kind audio_extraction --feed-url https://example.com/feed.xml \
  --episode-guid episode-42 --out-dir ./results

# Explicit enclosure provenance needs no feed selector.
extract-remote --kind audio_extraction \
  --enclosure-url https://cdn.example.com/episode.mp3 --out-dir ./results
```

Kind inference is intentionally narrow: archive extensions map to `image_note`
and video extensions map to `video`. Bare audio belongs to `transcript-remote`;
podcast `audio_extraction` requires RSS/enclosure provenance. `--frames` and
`--cadence` apply only to video. Feed selection accepts a GUID, an episode URL,
or both (the URL cross-checks the GUID). Title plus published date is the
last-resort fallback and cannot be combined with GUID/URL selectors.

The downloaded zip is untrusted. The client rejects traversal, symlinks,
duplicate/case-colliding paths, extra members, and asset size/hash mismatches. It
streams once into a sibling staging directory and renames only after complete
verification, so a failed download never leaves a partial final directory. Both
the streamed zip and declared assets are capped around the server's 2 GiB
archive ceiling to stop decompression/disk-exhaustion bundles.

### 7.4 HTTP API reference

| Method & path | Auth | Body / params | Returns |
|---------------|------|---------------|---------|
| `GET /health` | none | — | `{status, model, queued_or_running[]}` |
| `POST /jobs` | bearer | multipart: exactly one `file` or public HTTP(S) `url`; optional `diarize`, `language`, speaker hints, `detect_music` | transcript job status |
| `GET /jobs` | bearer | — | list of jobs |
| `GET /jobs/{id}` | bearer | — | status (`queued`, `running`, `done`, `error`) |
| `GET /jobs/{id}/result` | bearer | `format=txt\|srt\|vtt\|json` | rendered transcript (text/plain) |
| `DELETE /jobs/{id}` | bearer | — | cancel queued or delete terminal job; running is `409` |
| `POST /extractions` | bearer | multipart extraction form (below) | extraction status |
| `GET /extractions/{id}` | bearer | — | status; completed bundles survive restart |
| `GET /extractions/{id}/result` | bearer | — | canonical `ExtractionResult` JSON |
| `GET /extractions/{id}/bundle` | bearer | — | zip containing exactly `result.json` + declared assets |
| `DELETE /extractions/{id}` | bearer | — | cancel queued/delete error or durable done bundle; running/leased is `409` |

`POST /extractions` always needs `kind=video|image_note|audio_extraction`.
ASR-capable kinds accept optional `diarize`, `language`, `min_speakers`,
`max_speakers`, and `detect_music`. Kind-specific fields are:

- `video`: exactly one `url` or `file`; optional `frames=true` and `cadence_s`.
  Cadence must be at least 0.5 seconds.
- `image_note`: one uploaded archive in `file`; no URL.
- `audio_extraction`: either `enclosure_url`, or `feed_url` plus `episode_guid`,
  `episode_url`, or both. The complete `episode_title` + `episode_published` pair
  is a fallback used only without GUID/URL.

Queued status includes a 1-based `queue_position`. Extraction failures may add
machine-readable `error_reason` (for example `ambiguous` or `stale_selector`)
alongside the human-readable `error`. A `410` means a previously known bundle
was deleted, evicted, or lost; `404` means the ID is unknown.

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

# cancel a queued job or delete its terminal result
curl -X DELETE -H "Authorization: Bearer $TRANSCRIPT_TOKEN" \
     "$TRANSCRIPT_SERVER/jobs/ab12cd34ef56"
```

---

## 8. Output formats

| Format | Extension | Contents |
|--------|-----------|----------|
| `txt`  | `.txt` | Plain transcript, one line per segment, `SPEAKER_xx:` prefix when diarized. |
| `srt`  | `.srt` | SubRip subtitles, numbered cues, `HH:MM:SS,mmm` timestamps, `[SPEAKER_xx]` prefix. |
| `vtt`  | `.vtt` | WebVTT subtitles, `HH:MM:SS.mmm` timestamps. |
| `json` | `.json` | Full structured data: segments, words, timestamps, scores, speakers, language, meta. |

The JSON form is the canonical transcript output; the others are derived views.
The legacy JSON shape is frozen by an explicit serializer and deliberately
omits per-segment `music`, even when detection is requested. Opt-in music
metadata is added only for `--detect-music`, preserving default output bytes for
existing consumers. Per-segment `music` is available in extraction envelopes.

Extraction JSON is a separate envelope containing `kind`, canonical `text`,
`assets[]` (`key`, `sha256`, `size`, media type), modality-specific
`segments`/`frames`/`cards`, and `meta` provenance. `image_note` text uses
`## card N` blocks in archive order. Video frame OCR stays in `frames[]` and is
not mixed into spoken transcript text.

With `--detect-music`, segments that overlap classified music for at least half
their duration are marked and rendered with `[MUSIC]`. Metadata records:
`music_detection_requested`, `music_detection_succeeded`,
`music_detector_version`, `music_overlap_threshold` (`0.5`), and
`music_segments_flagged`. If the optional detector is absent or fails, the job
continues and records failure instead of silently changing the default path.

---

## 9. Configuration reference

### 9.1 CLI flags (`transcript`)

| Flag | Default | Meaning |
|------|---------|---------|
| `source` (positional, 1+) | — | Local file path(s) or http(s) URL(s). |
| `-f, --format` | `txt` | `txt` / `srt` / `vtt` / `json`. |
| `-o, --output` | stdout | Write to this file. |
| `--out-dir` | none | Batch/single output directory with numbered safe filenames. |
| `--model` | `large-v3` | Whisper model name (e.g. `medium`, `small`). |
| `--diarize` / `--no-diarize` | on | Speaker labels on/off. |
| `--language` | auto | Force a language code (e.g. `en`). |
| `--device` | auto | `cuda` or `cpu`. |
| `--compute-type` | auto | CTranslate2 compute type (e.g. `float16`, `int8`). |
| `--hf-token` | `$HF_TOKEN` | Hugging Face token for diarization. |
| `--min-speakers` / `--max-speakers` | none | Diarization hints. |
| `--batch-size` | 16 | ASR batch size. |
| `--no-align` | off | Skip word alignment (faster, coarser timing). |
| `--detect-music` | off | Explicitly run optional music classification/tagging. |
| `-v, --verbose` | off | Verbose logging / full tracebacks. |

### 9.2 Remote client flags

Both clients accept `--server`, `--token`, `--no-diarize`, `--language`,
`--min-speakers`, `--max-speakers`, `--detect-music`, `--poll`, `--timeout`, and
`--quiet`. `transcript-remote` additionally accepts `--format` and `--output`.
`extract-remote` accepts the kind/source selectors documented in §7.3 plus
`--out-dir`. Invalid speaker ranges and incompatible kind options are rejected
before network or file upload work starts.

### 9.3 `transcribe()` parameters

Mirror the CLI flags, plus: `work_dir` (where downloads/temp audio live),
`keep_audio` (don't delete temp files), and `engine` (reuse a warm
`TranscriptionEngine`). `detect_music=False` keeps music classification out of
the default pipeline.

### 9.4 Environment variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `HF_TOKEN` / `HUGGINGFACE_TOKEN` | server, engine | Diarization model auth. |
| `TRANSCRIPT_TOKEN` | server, client | Bearer auth token. |
| `TRANSCRIPT_SERVER` | client | Server base URL. |
| `TRANSCRIPT_HOST` / `TRANSCRIPT_PORT` / `TRANSCRIPT_MODEL` | launch scripts | Server bind + model. |
| `TRANSCRIPT_TOKEN_FILE` | launch scripts | Persisted generated token path. |
| `TRANSCRIPT_DATA_DIR` | server | Durable extraction root. |
| `TRANSCRIPT_OCR_MODEL_DIR` | server | Pinned OCR root containing `det/rec/cls`. |
| `TRANSCRIPT_OCR_ALLOW_DOWNLOAD=1` | server | Explicitly allow PaddleOCR first-run downloads. |
| `TRANSCRIPT_MAX_UPLOAD_BYTES` | server | Uploaded file-byte cap; default 8 GiB. |
| `TRANSCRIPT_MAX_QUEUE_SIZE` | server | Waiting-job cap; default 32. |
| `TRANSCRIPT_JOB_TTL_SECONDS` | server | Terminal in-memory transcript retention; default 86400. |
| `TRANSCRIPT_MAX_TERMINAL_JOBS` | server | Terminal transcript count cap; default 100. |
| `TRANSCRIPT_JANITOR_INTERVAL_SECONDS` | server | Cleanup sweep interval; default 900. |
| `TRANSCRIPT_MAX_CONCURRENT_BUNDLES` | server | Concurrent zip-build cap; default 8. |

---

## 10. Deployment

| Target | Asset | Notes |
|--------|-------|-------|
| Windows (5090 PC) | `deploy/run-server.ps1`, `deploy/run-server.bat`, `deploy/WINDOWS.md` | Owner-only persisted token, firewall, LAN IP. |
| Linux | `deploy/run-server.sh` | Owner-only persisted token + launch. |
| Linux (service) | `deploy/transcript-server.service` | Root-owned EnvironmentFile, auto-restart. |

To keep the server running unattended:
- **Windows:** NSSM service or a Task Scheduler "at startup" task (see
  `deploy/WINDOWS.md`).
- **Linux:** the provided systemd unit (`systemctl enable --now transcript-server`).

The shell launcher stores its generated token at
`${XDG_CONFIG_HOME:-~/.config}/transcript/server.token`; PowerShell uses Local
AppData. Set `TRANSCRIPT_TOKEN_FILE` to override either. Existing environment
tokens are never overwritten.

For systemd, copy `deploy/transcript-server.env.example` to
`/etc/transcript-server.env`, replace both tokens, and keep it mode `600`. The
unit reads this `EnvironmentFile` and passes `TRANSCRIPT_HOST`,
`TRANSCRIPT_PORT`, and `TRANSCRIPT_MODEL` to the server. Do not put secrets in
the unit or repository.

### Durable extraction data and cleanup

Completed extraction bundles are stored at
`$TRANSCRIPT_DATA_DIR/<job-id>/` (default
`~/.cache/transcript/extractions/<job-id>/`). Each contains immutable
`result.json`, declared assets, and a mutable side manifest. Bundles survive a
server restart and expire seven days after their last result/bundle access. A
janitor runs every 15 minutes by default; active read leases block deletion.
`DELETE /extractions/{id}` provides explicit cleanup.

Legacy terminal transcript jobs remain in memory for at most one day and 100
terminal records by default. Queued uploads are bounded by the 32-job queue;
their temporary files are removed on cancellation, completion, error, and stale
startup cleanup. Tune only after measuring workload with the variables in §9.4.

---

## 11. Performance & hardware

- **NVIDIA GPU (CUDA)** is the intended runtime. With `large-v3` + `float16`,
  transcription runs many times faster than realtime; diarization adds overhead
  but stays well below realtime on a modern GPU.
- **RTX 5090 (Blackwell, sm_120)** requires a CUDA 12.8+ PyTorch wheel, or you
  may see `no kernel image is available`.
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
  that enforces the token and its own request-body limit.
- API `url` fields accept only HTTP(S), resolve DNS, and reject non-global IP
  destinations (loopback, private/link-local, carrier-grade NAT, multicast, and
  reserved ranges). API callers cannot submit server-local file paths. Keep
  outbound firewall rules as defense in depth because media downloaders may
  follow redirects internally.
- Upload filenames are treated as untrusted. `TRANSCRIPT_MAX_UPLOAD_BYTES` caps
  file bytes; the ASGI body limit automatically adds 1 MiB for multipart
  framing. Configure a smaller reverse-proxy ingress cap if 8 GiB is unnecessary
  for your deployment.
- The launch scripts restrict persisted token files to the current owner. The
  PowerShell firewall rule targets only the **Private** profile. Keep `HF_TOKEN`,
  `TRANSCRIPT_TOKEN`, and systemd environment files out of source control.
- Jobs and extraction bundles have bounded queue/retention cleanup (§10).
  Downloaded bundle members are verified by the client before atomic publish;
  never bypass that check for an untrusted server.

---

## 13. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Required tool 'ffmpeg' was not found` | ffmpeg not installed / not on PATH | Install ffmpeg (§4.1); reopen the terminal. |
| `Speaker diarization needs a Hugging Face token` | `HF_TOKEN` unset or license not accepted | Set the token and accept the pyannote licenses (§4.4), or use `--no-diarize`. |
| `no kernel image is available for execution` | torch wheel too old for your GPU | Install a current cu128+ wheel (§4.2). |
| `compute_type float16 ... unsupported` on CPU | float16 on a CPU build | Handled automatically (retries int8); or pass `--compute-type int8`. |
| Mac can't reach the server | firewall / network profile / wrong IP | Set LAN to Private, run PowerShell as admin once, `curl /health` with the printed IP. |
| `401 unauthorized` from the server | missing/incorrect token | Set `--token` / `$TRANSCRIPT_TOKEN` to match the server. |
| OCR assets have empty text | OCR weights unavailable | Set `TRANSCRIPT_OCR_MODEL_DIR` or explicitly allow download (§4.5); inspect extraction `meta.ocr_warning`. |
| Music flags stay empty | detector missing/failed | Install `.[music]`, pass `--detect-music`, and inspect music metadata (§8). |
| extraction returns `410` | durable bundle was evicted/deleted | Resubmit; copy verified bundles elsewhere when they need retention beyond seven days. |
| submission says queue is full | 32 waiting jobs already queued | Wait/cancel queued work; increase `TRANSCRIPT_MAX_QUEUE_SIZE` only if disk capacity allows. |
| yt-dlp download fails | site change / no network / age-gate | Update `yt-dlp` (`pip install -U yt-dlp`); check the URL. |
| Word timestamps missing | alignment failed for that language | Non-fatal; transcript still produced. Check `-v` logs. |

---

## 14. Development

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest -q   # fast suite; no GPU/model download
ruff check .
```

CI runs Ruff and the fast suite on Python 3.10 and 3.12. A real media check is
available but skipped by default:

```bash
# Generate a one-second ffmpeg input and run WhisperX (tiny model by default).
TRANSCRIPT_REAL_MEDIA_SMOKE=1 PYTHONPATH=src \
  pytest -q tests/test_real_media_smoke.py

# Also exercise yt-dlp using a media URL you are authorized to download.
TRANSCRIPT_REAL_MEDIA_SMOKE=1 TRANSCRIPT_SMOKE_URL=https://example/media \
  TRANSCRIPT_SMOKE_MODEL=small PYTHONPATH=src \
  pytest -q tests/test_real_media_smoke.py
```

Conventions:
- Keep `torch`/`whisperx` imports lazy (inside functions in `engine.py`).
- Keep legacy transcript serialization separate from extraction envelopes.
- Line length 100; `ruff` configured in `pyproject.toml`.
