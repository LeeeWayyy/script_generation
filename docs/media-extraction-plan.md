# Media Extraction Service — Plan (Phase 3, processing side)

Evolve `transcript` from an **ASR-only** service into a modality-aware
**"media → candidate text (+ assets) + a provenance recipe"** service consumed by
a downstream LLM-wiki vault (DailyNotes). Reviewed across three rounds (codex +
gemini + claude per round); this folds all three.

> *Code lives under `src/transcript/`; cited line numbers are indicative and drift
> as edits land — bind to the named symbol (`Worker.run`'s `meta.update`,
> `transcribe()`, `render(job.transcript, …)`, the `bestaudio/best` selector,
> `extract_audio`), not the number. This matters most for the byte-golden test,
> whose entire value is pinning exact production behavior.*

**Hard boundaries (non-negotiable):**
- **Vault-agnostic.** Returns text + asset files + a `meta` recipe; knows nothing
  about sidecars/citations/dedup/wiki. Returns **candidate** text — the consumer
  commits + hashes to confer canonicality.
- **ASR path byte-unchanged.** The existing audio `transcript-remote -f json`
  output must not change (the consumer hashes it). A separate result envelope is
  **necessary but not sufficient** — byte identity also depends on the legacy
  route, dataclass field order, JSON literals (`indent=2, ensure_ascii=False`),
  and **`meta` insertion order**. The guarantee = separate envelope **+** untouched
  legacy rendering **+ byte-golden tests** (below). The existing `test_formats.py`
  suite passing is NOT that guarantee — those tests build `Transcript` directly and
  never exercise `transcribe()`/`Worker.run` meta-stamping, the exact surface this
  change touches; **the real acceptance gate is the byte-golden production-path
  test** (§0), not the current `test_formats.py` count.
- **Thin client stays `requests`-only.** Heavy deps (PaddleOCR, ffmpeg, feed
  parser) are **server-side only**, lazy-imported inside functions (never at
  package/module top — mirrors `engine.py`'s whisperx import), so the client/tests
  never touch them. The new client uses only `requests` + stdlib `zipfile`.

## Current state (done)

`transcribe(url|file) -> Transcript{segments, language, meta}`; async `/jobs`
server (in-memory `JobStore` dict, `server.py:71`; temp work dir deleted in the
worker `finally`, `server.py:145-148`); ASR = WhisperX + pyannote. `meta` is built
by ordered `.update()` calls (engine.py:150, `__init__.py:105/143`, server.py:138).
`/jobs/{id}/result?format=json` renders `render(transcript,"json")` =
`json.dumps(transcript.to_dict(), indent=2, ensure_ascii=False)` (formats.py:60).

## 0. Result envelope + the two byte-break leaks (do FIRST)

`Transcript.to_dict()` is `asdict`; adding `cards`/`frames` would change ASR JSON.
Introduce a **separate `ExtractionResult`** envelope:

```
ExtractionResult:
  kind: "video" | "image_note" | "audio_extraction"  # NOT plain "audio" (legacy ASR)
  text: str                      # candidate canonical-text (named renderer, below)
  segments: list[Segment] | None # audio/video (existing shape)
  frames:   list[Frame]   | None # video (§B)
  cards:    list[Card]    | None # image_note (§A)
  language: str | None
  assets:   list[AssetRef]       # opaque job-relative keys + integrity (§Interface)
  meta:     dict                 # the provenance recipe
```

**`AssetRef` carries integrity, not just a key:** `{key, sha256, size,
media_type}` for **every** asset type (cards have `image_sha256`, but frames must
too — `image_ref` alone lets a truncated/corrupt bundle still "match" the
envelope). The consumer verifies `sha256`/`size` after unzip. **Server-side `key`
invariant:** unique, POSIX-relative, non-empty, not `result.json`, not absolute, no
`..` component — so duplicate zip members / path aliases can't make verification
ambiguous (independent of the client's own zip-slip rejection).

- **Routing — close leak #1 (audio via envelope), as a MUST.** Plain **ASR `audio`
  is not an envelope kind**: `transcript-remote` (and `extract-remote --kind audio`,
  if offered) hits the *legacy* `/jobs/{id}/result?format=json` →
  `render(Transcript)` path and returns `Transcript` JSON — full stop. Extraction
  kinds (`video`/`image_note`/**`audio_extraction`**) get a **separate route**
  (e.g. `/extractions/{id}/result`); the legacy `/jobs/{id}/result` stays
  `Transcript`-only and is **never** reused for the envelope. The new route does
  NOT go through `formats.render`/`to_json`. **Podcast audio uses
  `kind: audio_extraction`** (NOT plain ASR) precisely because it must carry
  RSS-resolution provenance (`feed_url`/`episode_guid`/`resolution_source`, §C) on
  `ExtractionResult.meta` — that provenance has no byte-safe home on
  `Transcript.meta`, so RSS-resolved audio is an extraction, while a bare-URL/file
  ASR job stays legacy. (Distinct kind + separate route makes the boundary
  structural — a future "cleanup" can't route legacy ASR through the new dataclass.)
- **`meta` confinement — close leak #2 (shared meta-stamp).** The audio meta-stamp
  join point is **`server.py:138`** (`job.transcript.meta.update({"job_id":…,
  "server_version":…})`) — a single line that today *unconditionally* mutates
  `job.transcript.meta`. **That line must branch on `kind`:** audio →
  `Transcript.meta`; envelope kinds → `ExtractionResult.meta`. §C's podcast fields
  (and OCR/frame fields) land on **`ExtractionResult.meta`** and must never reach
  the `server.py:138` audio path — do NOT wire them into the existing
  `transcribe()`/`Worker.run` merge.
- **Byte-golden regression (step-0 deliverable).** `test_formats.py` builds
  `Transcript(...)` directly (lines 9-22) and never runs `transcribe()`/
  `Worker.run`, so it cannot catch a `meta`-key regression. The test must **drive
  the real production functions** (`transcribe()` then the `Worker.run` merge) with
  **monkeypatched** `resolve_source`/`extract_audio`/engine/version-helpers — not a
  hand-authored meta dict — and **compare exact bytes (incl. key order)** of
  `to_json`. A hand replay can silently encode the wrong key order. The sites it
  must exercise are the real four-site insertion order —
  `engine.run`'s `meta.update` → `transcribe()`'s first `meta.update` →
  `transcribe()`'s URL-recipe `meta.update` (URL only) → `Worker.run`'s
  `meta.update` — and pins **both the URL and local-file shapes** (they differ;
  local-file skips the download recipe). This list is the **resulting `meta` key
  order** (also the execution order — the engine update fires inside `eng.run()`
  *before* the `transcribe()`-level updates mutate the returned object), not
  source-line order; **cite the symbols, not line numbers** (which drift). A
  hand-authored meta in the wrong key order would lock in non-production bytes.
  The assertion is on **exact bytes from the public `/jobs/{id}/result?format=json`
  route** (the actual contract surface), not only `to_json()`. This guard must
  exist **before** the first `meta`-mutating feature (podcast) and before the
  `Worker.run` kind-branch ships.
- **Per-modality `text` = a named renderer with tests, not prose.** image_note →
  each card's `ocr_text`, **1-based `## card N`** (pin 1-based explicitly; `index`
  is 0-based), blank-line-separated, OCR line-breaks preserved, with the **literal
  rules pinned**: zero cards → `""`; each card is `## card N\n<ocr_text>`; an
  empty-OCR card renders `## card N\n` (no body); cards joined by `\n\n`; the
  rendered text **ends with a single trailing `\n`** (cf. `to_txt`'s `+ "\n"`,
  formats.py:37). Golden fixtures pin zero/one/many-card + trailing-newline cases. video → `formats.to_txt()` semantics (speaker prefixes + trailing
  `\n`); frame OCR kept **separate** in `frames[].ocr_text`, never merged into
  audio text. **`audio_extraction.text = NFC(formats.to_txt(transcript))`,
  computed in the NEW extraction renderer** (speaker prefixes, blank segments
  skipped, trailing `\n`, then NFC) — this is **not** the legacy `transcript-remote
  -f txt/json` code path. **NFC is an extraction-renderer-only rule** applied
  identically across the three extraction `text` renderers; it is **NOT** applied
  inside `formats.py` / `Transcript` / `Segment` / the legacy `/jobs/{id}/result`
  route (normalizing there would change byte-stable ASR output). Divergent
  normalization across the *extraction* renderers is a silent hash split, hence
  "identical"; the legacy path is simply excluded. `srt`/`vtt` invalid for image_note → clear
  error. The consumer hashes `text`, so this is contract. **`ExtractionResult` JSON
  serialization is ALSO pinned** by **one explicit serializer** (fixed field order,
  explicit `None`-vs-omit-vs-`[]` inclusion policy, float formatting, `indent`,
  NFC — do NOT rely on FastAPI/Pydantic defaults): the durable `result.json` in the
  bundle must be byte-identical to the `/result` route, so its bytes are an
  operational contract even though DailyNotes hashes `text` + committed artifacts,
  not the envelope JSON.
- **`AssetRef` = opaque job-relative key** (`assets/card-000.jpg`,
  `assets/frame-000123.jpg`) — neither URL nor server path. The consumer decides
  commit location + citation grammar.

## A. OCR (`image_note` cards + video-frame text)

- **Engine:** PaddleOCR (strong CJK), **lazy-imported** in a new `ocr.py`. **Model
  weights pre-fetched at deploy to a pinned cache path** — first-run download is a
  hidden network dependency that can fail mid-job; fail clearly (cf. yt-dlp's
  friendly `RuntimeError`, ingest.py:61-64), don't surprise-download.
- **Determinism (softened):** deterministic only under a pinned
  engine+model+params+device — NOT byte-stable across CPU/GPU, Paddle minor
  versions, or fp16 nondeterminism. Mitigation = the recorded recipe
  (`ocr_engine@version`, `ocr_model`, `ocr_params`). **`ocr_params` must include
  the image-preprocessing rules** — EXIF-orientation handling, resize/
  `det_limit_side_len`, colorspace, alpha flattening, angle-classifier, `lang` —
  because OCR sees *decoded pixels*, so the same original bytes yield different
  `ocr_text` across libraries/devices without these recorded.
- **Reading order is pinned, not PaddleOCR's raw output order** (which varies with
  layout/model/orientation). Define the `blocks[]` sort (e.g. top-to-bottom,
  then left-to-right, with a column/threshold rule for multi-column + vertical CJK
  + rotated photos) and state that `ocr_text` is generated **from that ordering**.
  `cards[].ocr_text` is an **observation**; the consumer hashes the returned
  artifact, recipe explains divergence.
- **`cards[]` shape:** `{index, ocr_text, image_ref, source_filename,
  image_sha256, width, height, confidence, blocks?}`. Per-card `confidence` (not
  top-level) with a **chosen aggregation rule** (pick one — e.g. arithmetic mean of
  non-zero line scores — don't defer it; the field is otherwise unspecified).
  Optional `blocks[]` (line text + bbox + per-line score, in the pinned reading
  order above) retained for OCR debugging + later citation UI; keep `text` simple.
  **`image_sha256` is over the ORIGINAL bytes** (for an archive input, the
  *extracted member's* original bytes — not a re-encode, and not the enclosing
  zip/tar); **`source_filename` is the bare basename** (post-flat-extract; drives
  nothing — ordering is by §A's sort). Also include the **original archive member
  path as a sanitized, non-key `source_member` observation** (never used for asset
  paths or ordering) so manual-export debugging can answer "which source image made
  this card."
- **Card ordering = UTF-8-byte sort over NFC-normalized basenames** (codepoint and
  byte order differ for non-ASCII — pick this one concretely; normalize to NFC
  *before* both sorting and collision detection): it drives `index` which drives
  the hashed `text`, and a locale/case-sensitive sort drifts between a macOS client
  and a Linux server.
  **Define basename-collision behavior:** "flat-extract by basename" (below) means
  two members `a/1.jpg` and `b/1.jpg` collide — **reject** (or deterministically
  suffix), never silently overwrite, since a lost member shifts every `index` and
  thus the hashed `text`.

## B. Video frame extraction + OCR

- **Acquisition — close the audio-stream leak.** The current path downloads
  `-f bestaudio/best` (ingest.py:46) and `audio.extract_audio` always does
  `-vn -ar 16000 -ac 1 -acodec pcm_s16le` (audio.py:31) — so the *container* it
  decodes from doesn't affect ASR bytes. **BUT** a muxed `best`/`bestvideo+
  bestaudio` may carry a **different audio stream** than `bestaudio/best` → a
  different decoded WAV → a silently different transcript for the same URL. Pin:
  when `--frames` is on, **ASR audio is still taken from the same `bestaudio`
  stream** (download `bestaudio` for ASR + a capped video stream for frames), so a
  video job and an audio job for one URL yield the **same** transcript. Extract
  frames from the video media and **copy them into the durable asset dir before the
  existing temp-dir cleanup fires** (`__init__.py:146-147` / server.py:147-148 —
  the work-dir lifecycle changes for frame jobs). A frame job downloads **two
  logical streams**, so record **both** `selected_audio_format` (the ASR
  `bestaudio` stream) and `selected_video_format` (the frame stream) on
  `ExtractionResult.meta`; the legacy singular `selected_format` stays only on the
  old ASR path's `Transcript.meta`. **Local/uploaded video** has no separate
  `bestaudio` download — ASR already decodes from the same local media via
  `extract_audio()` (`__init__.py:~86`); frame extraction **reuses that same local
  source** and introduces no remux/transcode before ASR.
- **No cross-modal timestamp-alignment guarantee.** Because frames come from a
  separate video stream, **frame timecodes are on the video stream clock and
  transcript segment times are on the ASR audio stream clock** — start offsets /
  ads / edits can desync them. Frame OCR lives in `frames[]`, separate from the
  audio `text`, so this is acceptable; state it so no consumer assumes alignment.
- **Two-stream download must not collide.** The current path writes
  `%(id)s.%(ext)s` + `<id>.info.json` (`ingest.py:~39/51`); a separate `bestaudio`
  (ASR) + capped-video (frames) download in one work dir can overwrite/ambiguate
  the info JSON. Use **separate subdirs or distinct output templates** per stream,
  then record both formats.
- **Policy — fixed cadence DEFAULT** (reproducible); scene-detect **opt-in**.
  Record cadence_s, exact `ffmpeg` select/scale/colorspace, timestamp rounding,
  neutral frame naming, pHash impl+size+distance, sort order, AND the **frame
  encoding params** (output format, JPEG/WebP quality, pixel format, scale
  algorithm, EXIF handling, re-encode-vs-copy) — asset hashes are only useful if
  the recipe explains why they changed. **Frame cap / cadence floor** for long
  video.
- **`frames[]` + `frames.json` manifest:** `{frame_id (the ordinal N, pinned),
  timecode, image_ref, ocr_text}`. **Pin `timecode`'s format** (seconds as a number
  vs fixed-decimal string vs ffmpeg timestamp, + rounding precision) and — if
  `blocks[]` is retained — its **bbox coordinate system, units, and order**;
  leaving these implicit risks downstream citation drift. Per-frame OCR **reuses
  the §A card OCR shape** (confidence + optional `blocks[]`) so video OCR is as
  auditable as image_note OCR — or explicitly narrows it with a stated reason.
  Integrity is **single-sourced** — `image_ref`
  resolves to its `AssetRef` (which carries `sha256`/`size`); don't duplicate a
  separate `frames[].image_sha256` (two hashes invite a mismatch with no defined
  winner). Frame *selection* is reproducible only against the recorded
  policy+ffmpeg version; `frames[].ocr_text` is an **observation** — never recipe.

## C. Podcast / RSS resolution

- **Feed parse PRIMARY (`feedparser`, lazy), yt-dlp info.json fallback:** yt-dlp's
  `%(id)s` does not reliably surface the RSS `<guid>`. Feed parse yields
  `feed_url`, `episode_guid`, `enclosure_url`, `published`; yt-dlp downloads — and
  is handed the **selected `enclosure_url`**, not the feed/episode-page URL (given a
  page URL it may pick a different asset than the feedparser-selected enclosure,
  esp. for multi-enclosure or platform pages).
- **Episode selection required, with a precedence contract.** A raw feed URL
  doesn't identify an episode — require `episode_guid` or `episode_url`. Pin the
  exact behavior: when **both** are supplied, `episode_guid` wins (and a mismatch
  with `episode_url` is a structured error); when GUIDs are **missing**, fall to
  `episode_url` then `(title, published)`; when GUIDs are **duplicated** within the
  feed, or `episode_url` matches **multiple** entries after canonicalization →
  structured **ambiguous** error. Distinguish **ambiguous** from **stale selector**
  (selector matches *zero* entries — the GUID-instability case) so the consumer can
  tell them apart. **Fallback with no provable feed identity:** if feed parse fails
  / the input is not a feed and the yt-dlp fallback cannot prove an
  `episode_guid`/selected enclosure, it must EITHER require a user-supplied
  enclosure URL (recorded `resolution_source=user_supplied`) OR return a structured
  "feed identity unavailable" error — never silently mint weak podcast provenance.
- **Matching is NOT raw URL equality.** Enclosure URLs are tracking-wrapped and
  redirect through CDNs with per-request params, so the downloaded URL rarely
  string-equals the feed's `<enclosure>`. **Prefer a stabler key first:**
  `episode_guid` plus enclosure `length`+`type` where present. URL canonicalization
  is a tiebreak, **defined to the byte** (lower-case scheme+host, strip a leading
  `www.`, http→https only when the feed declares it, drop the fragment, collapse a
  trailing slash, decode safe percent-escapes, unwrap known feedburner/tracking
  redirectors) so `episode_url` comparison can't false-ambiguate; and **following
  redirects to a "final host+path" is best-effort only** — it's a network side-effect that can
  time out or land on a region-varying CDN host, so matching correctness must not
  depend on it. **Multi-enclosure** entries (mp3/m4a/ogg): prefer audio MIME, fail
  if multiple audio enclosures remain. When the feed `<enclosure length>` is
  present, **verify it against the downloaded file size and record both** — but
  fatal **only when the length is authoritative**: a complete (non-ranged, fully
  downloaded, non-redirected-to-transformed) response. Stale/missing lengths and
  CDN range/transform responses are recorded as observations, not failures (else
  valid feeds fail for provenance theater). Record the redirect chain +
  `Content-Length` + final size so the trust decision is auditable.
  Record `resolution_source`
  (`feed_parse`|`yt-dlp_info_json`|`user_supplied`) + **both** the feed
  `enclosure_url` and the final post-redirect download URL **on
  `ExtractionResult.meta`**, each tagged **observation** (not a stable recipe
  field). *(No "known-feed" warning — this service is vault-agnostic and job-local;
  it holds no cross-job feed history, so GUID-drift detection belongs to the
  stateful consumer, not here.)*

## Interface / dispatch / asset delivery

- **Durable assets + the completed result, not just an index.** Add a **durable
  per-job asset directory** (don't delete on worker finish, `server.py:145-148`).
  `JobStore` is in-memory and `/result` reads it, so on restart the assets survive
  but `text`/`assets[]`/`meta` are gone — the client can't use the orphaned bundle.
  So **persist the completed `ExtractionResult` JSON (+ asset manifest) beside the
  asset dir**, written atomically on completion; the `created_at`/`last_access`
  TTL timestamps live **in that on-disk manifest** (not only the in-memory `Job`).
  A startup scan rebuilds the index from those manifests. (Acceptable alternative:
  explicitly declare completed jobs **unrecoverable after restart** and have the
  janitor eagerly GC orphaned dirs — but then say so.) Without one of these:
  guaranteed disk leak + un-fetchable results.
- **Data-model additions (call them out).** Eviction has nothing to act on today:
  `Job` (server.py:37-51) has no timestamp and there is only the single `Worker`
  thread. So this adds `created_at`/`last_access` (persisted in the on-disk
  manifest, above) AND a **new background janitor thread** distinct from `Worker`,
  on a stated cadence (e.g. every ~15 min). **The mutable TTL fields
  (`created_at`/`last_access`) live in a side manifest, NOT inside `result.json`** —
  `result.json` is the immutable, hashed bundle member, so a fetch bumping
  `last_access` must not rewrite its bytes.
- **Eviction race + fetch contract.** TTL eviction runs concurrently with `GET
  /extractions/{id}/bundle` (the extraction route everywhere — never grow the
  legacy `/jobs/*` API). Rules: never evict a running job; **bump last-access (and
  thus TTL) on a fetch**, hold a lease (or evict-by-rename-then-delete) so a sweep
  can't unlink a dir mid-stream; assets are **immutable once `status==done`** so a
  dropped bundle download is safely idempotent on retry. **Atomic bundle
  readiness + publish ordering:** build assets + write the manifest/result under a
  **staging dir**, **rename into place**, and only **then** publish `status=done`
  (the current worker sets `done` in memory right after the meta-stamp,
  `server.py:~139` — if `done` precedes the durable write a client can observe
  success before `/bundle` is finalized). Define the **`410 Gone` vs `404`**
  contract for "`/result` lists `assets[]` but the bundle was TTL-evicted / lost on
  restart." **`/result` and `/bundle` share the read-lease:** a `GET /result` that
  returns `assets[]` bumps `last_access` too (or the doc states `/result` alone is
  not a reservation) — else a client reads `/result`, then loses the bundle to TTL
  before fetching `/bundle`. **Restart semantics are explicit:** completed
  extractions (manifest+`result.json` already renamed into place) survive restart
  and the startup scan re-serves them; running/staging extractions do not survive
  and return `404`/structured-failed after cleanup.
- **Errored jobs:** an extraction that fails after partially producing assets
  persists **no bundle** and the staging dir is GC'd (the job record carries the
  error; no half-bundle is ever exposed). Pick this over an error-manifest bundle.
- **Crash/staging cleanup.** If the process dies mid-job before the atomic rename,
  a staging/running dir is left with no completed manifest. The **startup janitor
  GCs stale staging/running dirs that have no completed-result manifest** — else
  the durable asset disk leaks on crashes even though normal TTL is handled.
- **Delivery = one zip bundle, on the separate extraction route.** The extraction
  result JSON is served from the **new** route (e.g. `/extractions/{id}/result`,
  §0 — never the legacy `/jobs/{id}/result`); `…/extractions/{id}/bundle` streams a
  zip; the client unzips via stdlib `zipfile`. **The zip contains the immutable
  `result.json` (same bytes as the `/result` endpoint) at a fixed top-level path,
  plus every asset at its exact `AssetRef.key`** — so manual archiving + retry are
  unambiguous and self-describing. **Zip determinism is NOT required** (the consumer
  hashes `text` + per-asset `sha256`, not the zip) — state this so nobody
  over-invests in reproducible zips.
- **Auth on the new endpoints.** `/bundle` and the new envelope `/result` route
  MUST carry `Depends(auth)` (server.py:178-183) — an unauthenticated `/bundle`
  leaks user media.
- **Serve extractions from a separate persisted `ExtractionRecord`, not a
  reconstructed `Job`.** `Job.public()` reads `self.transcript.speakers/.language/
  .segments` (server.py:53-67), which only makes sense for ASR. Rather than overload
  `job.transcript` and branch `public()` on kind, completed extractions are served
  from their own persisted record (the durable `result.json` + manifest, above) via
  the extraction routes — so a startup scan answers `/extractions/{id}` and
  `…/result` without touching the legacy `Job`/`Transcript` shape at all.
- **Client = a NEW `extract-remote` console entry** (`pyproject.toml`: add
  `extract-remote = "transcript.extract_remote:main"` beside the frozen
  `transcript-remote`), with shared HTTP/poll logic factored into a **named helper
  module** both import. `extract-remote --kind {video,audio_extraction,image_note}
  [--frames] [--out-dir <dir>] <url|file|bundle>`. **No `--kind audio` alias** —
  `audio_extraction` is spelled explicitly, so "audio" never names two commands
  that hash different bytes (plain ASR stays `transcript-remote`). On unzip it
  **verifies each `AssetRef`'s
  `sha256`/`size`** and **rejects absolute / `..` members** before writing —
  cheap defense so a compromised/misconfigured server can't write outside
  `--out-dir`.
- **Modality dispatch:** explicit `--kind` wins; else sniff by extension/mimetype
  (hard-fail on ambiguity). `--frames` is orthogonal to `--kind video`.
- **Manual-export input = a zip/tar bundle** (the API takes one `UploadFile`,
  server.py:195). **Extract untrusted archives safely — security must:** reject
  absolute/`..` members (zip-slip / tar path-traversal); reject **symlinks AND tar
  hardlinks + special files** (block/char devices, FIFOs — a separate traversal/DoS
  vector); flat-extract by basename only (collision rule per §A). **Cap
  decompression** — per-archive uncompressed-size + member-count limits — or an
  untrusted bundle (zip bomb) fills the now-non-self-cleaning durable asset disk.
  Card `index` is driven by the §A ordering rule (UTF-8-byte sort over
  NFC-normalized basenames). This is the supported contract; live RedNote fetch is
  best-effort.

## Provenance recipe (`meta`)

- Reuse `_pkg_version`/`_ffmpeg_version`; all best-effort → `null` when unknown;
  **all on `ExtractionResult.meta`**. OCR: `ocr_engine@version`, `ocr_model`,
  `ocr_params`. Frames: `frame_policy{method, cadence_s|threshold, phash,
  dedup_distance}`, `ffmpeg_version`, `frame_count`, `selected_audio_format` +
  `selected_video_format`. Podcast (on the `audio_extraction` envelope):
  `feed_url`/`episode_guid`/`enclosure_url`/`published`/`resolution_source`.
- **Mark each field recipe (stable) vs observation (varies).** OCR/frame outputs
  are observations: the consumer hashes the actual returned artifact for
  provenance and uses the recipe to *explain* why a re-run differs — it does not
  try to hash a "stable subset."

## Non-goals / constraints

- No vault awareness. ASR path byte-unchanged (separate envelope + untouched legacy
  route + byte-golden meta test). Heavy deps server-side + lazy; client
  `requests`+`zipfile` only. Live social fetch best-effort; the archive
  manual-export path is the contract.

## Sequencing

0a. **Byte-stability guard + route/client scaffolding FIRST:** the result envelope,
   the two leak fixes, the new extract route/client with zero behavior change to
   legacy ASR, and the byte-golden `meta` test (asserted on the `/jobs/{id}/result`
   bytes). Prove the ASR path didn't move before anything else lands.
0b. **Then the durable asset lifecycle** (durability/persisted result/eviction/auth/
   staging/janitor). Splitting 0a from 0b keeps the large infra change from landing
   before the byte guard exists. The byte-stable boundary must be tested before
   anything that mutates `meta`.
1. **Podcast** — **hard-gated on the URL *and* local-file byte-golden tests passing
   first** (podcast provenance is exactly the data that could leak into
   `Transcript.meta`, so the guard must exist before any RSS work). The small safe
   first slice is surfacing yt-dlp `info.json` metadata; the **feed parse + episode
   selection** (§C) is its own larger step.
2. **OCR + `image_note`/cards + the archive manual-export input.**
3. **Video frame extraction + OCR** (acquisition change + audio-stream pinning) —
   last.
