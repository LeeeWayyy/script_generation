# Initial Windows reference calibration

**Accuracy acceptance has not passed.** All 40 selected videos generated
successfully twice, all 80 outputs passed structural and ScriptViewing parser
checks, and all 40 raw/readable outputs matched exactly between independent
inference passes. This establishes repeatability in the same server process,
not correctness of every spoken word, speaker, or timestamp.

The corpus contains 20 English and 20 Japanese inputs, about 12.5 hours total.
Audio alone was uploaded to the live Windows API with alignment and diarization
enabled. Creator captions stayed outside generation. `frozen-inputs.json` pins
the input hashes; `full-results.json` retains every score, job identifier, output
hash, structural assessment, parser result, and repeat comparison.

## Reference agreement

These are diagnostic comparisons, **not measured ground-truth ASR error rates**.
All selected cases remain in these aggregates, including flawed references.
English uses normalized WER; Japanese uses normalized CER. Kana/kanji spelling
differences, caption edits and omitted fillers can contribute to disagreements.

| Set | Videos | Reference-weighted error | Flagged reference cases |
|---|---:|---:|---:|
| English calibration | 14 | 6.53% WER | 1 |
| English holdout | 6 | 4.73% WER | 0 |
| Japanese calibration | 14 | 16.60% CER | 1 |
| Japanese holdout | 6 | 22.84% CER | 0 |

No generated transcript exactly matched its normalized reference. The Japanese
disaster case has only 203 reference characters, ending around 402 seconds of an
807-second video. It is unsuitable as a full-video accuracy reference and needs
replacement. The business reference contains suspect wording. Additional cooking
and computing calibration comparisons include apparent caption typos and wording
differences. A zero flag count does not establish reference quality.

## Fixes and rejected trials

- Fixed English scoring to equate spoken/written numbers while retaining currency
  and percent units. The original strict scores remain available. This is a
  measurement fix, not improved generation.
- Added timestamp-coverage warnings so partial creator captions are not presented
  without qualification as full-video references.
- Beam 10 matched beam 5 on the pilot calibration cases; no benefit established.
- Native decoding without previous-text context worsened English calibration and
  Japanese holdout agreement; rejected. Context-enabled sequential decoding in
  the WhisperX subclass produced severe repetition; rejected.
- Lowering speech detection from 0.5 to 0.35 changed Japanese conversation CER
  from 24.51% to 26.22% on one case and 19.04% to 18.66% on another; rejected.

The production decoder remains unchanged. Nine focused benchmark tests passed on
both Mac and Windows. The live server is healthy and idle; the earlier original
YouTube URL job remains available. Transient 403s during corpus download recovered
with fresh extraction retries; this does not prove that all future URLs will work.

## Outstanding acceptance work

The generation disagreements still need acoustic adjudication, and incomplete or
suspect references need replacement/review before they can define accuracy gates.
Speaker identity, word timing, sentence/expression boundaries and the spoken-only
requirement remain unverified. Twenty first-pass files contain some unknown word
speakers; 26 contain short rows flagged for review. Reproducibility and parser
success do not waive these requirements. No benchmark output was imported into
the user's application library.
