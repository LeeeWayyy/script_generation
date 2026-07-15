# Media Extraction Decisions

This record captures the constraints behind the extraction service. The implementation and
user-facing API live in `src/transcript/` and `docs/DOCUMENTATION.md`; this file is intentionally
limited to decisions that future changes must preserve.

## 0. Result contracts

- Legacy transcription remains `Transcript` rendered through `formats.py` and `/jobs/*`.
  Its JSON bytes are regression-tested and must not acquire extraction-only fields.
- `video`, `image_note`, and `audio_extraction` use the separate `ExtractionResult` envelope and
  `/extractions/*` routes.
- Extraction text is candidate text. This service does not know about downstream vaults,
  citations, deduplication, or canonical commits.
- `result.json` is serialized once by `extraction.serialize`; the result route and bundle contain
  those exact bytes.
- Every asset has a unique POSIX-relative key plus SHA-256, size, and media type. The client
  verifies the exact declared member set before atomically publishing a bundle.

## A. Image notes and OCR

- The supported input is an uploaded zip/tar manual export. Archive members are untrusted:
  traversal, links, special files, ambiguous names, excessive members, and excessive expanded
  data are rejected.
- Cards are ordered by UTF-8 bytes of NFC-normalized basenames. Flat-extraction collisions are
  errors because ordering affects candidate text.
- Original image bytes are preserved as assets. OCR is an observation over decoded pixels; its
  engine version, model, preprocessing, and parameters are recorded as provenance.
- OCR may degrade to empty text without discarding safe image assets. The envelope records the
  warning and whether OCR succeeded.

## B. Video frames

- ASR always uses the same best-audio path as legacy transcription. URL frame extraction uses a
  separate, resolution-capped video stream so enabling frames cannot silently change transcript
  audio.
- Frame selection is fixed-cadence and bounded. The exact ffmpeg selector, scale, encoding,
  timestamp source, rounding, and cap are recorded in `frame_policy`.
- Frame timestamps use the video stream clock; transcript timestamps use the ASR audio clock.
  Consumers must not assume cross-stream synchronization.
- Frame OCR remains in `frames[]` and is never merged into spoken transcript text.

## C. Podcast resolution

- `audio_extraction` requires either a feed plus an episode selector or an explicit enclosure URL.
  Bare audio remains on the legacy transcript route.
- Feed selection prefers GUID, cross-checks an optional episode URL, and otherwise uses an exact
  episode URL or title/published pair. Missing, stale, ambiguous, and conflicting selectors are
  distinct errors.
- Enclosures are fetched by the guarded, size-bounded downloader. Feed URL, selected enclosure,
  redirects, declared length, response length, downloaded size, and selection source are recorded
  as observations.
- An authoritative enclosure-length mismatch fails before ASR. Network failure never falls back
  to an opaque downloader that bypasses the enclosure safety policy.

## Interface and storage

- A single serial worker owns warm ASR and OCR engines. Add concurrency only after measurements
  show the GPU or queue needs it.
- Completed extractions publish through staging plus atomic rename, then become `done`. Incomplete
  staging data is never served.
- `result.json` and assets are immutable. A mutable side manifest tracks last access for TTL
  eviction.
- Result and bundle reads hold leases so concurrent deletion cannot remove active data. Known
  deleted/evicted results return `410`; unknown IDs return `404`.
- The thin clients remain `requests` plus the standard library. Heavy ML, OCR, archive processing,
  and media tools stay server-side and lazily imported.

## Security boundary

- Remote routes require bearer authentication before request bodies are parsed. An explicitly
  open server is suitable only for a trusted, firewalled network.
- User-supplied fetch URLs must be HTTP(S) and initially resolve only to globally routable
  addresses. Downloader redirects are additionally contained by deployment egress rules.
- Upload, remote-download, decoded-image, archive-expansion, frame-count, asset-byte, queue, and
  retention limits bound resource use.

## Non-goals

No scene detection, pHash deduplication, live social-site scraping contract, live transcription,
vault integration, or reproducible zip bytes. Add one only when a measured consumer need justifies
the extra behavior.
