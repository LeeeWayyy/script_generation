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

## Scoring v3 and verbatim-prompt trial

Scoring v3 (`reference-scores-v3.json`, same generated outputs) removes caption
annotations and speaker labels, and treats fillers/backchannels as optional (a
product decision: they are not required transcript content). It also adds a
full-coverage aggregate. This is a measurement change only.

| Set | v2 all cases | v3 all cases | v3 full-coverage references |
|---|---:|---:|---:|
| English calibration | 6.53% WER | 5.73% WER | 5.73% WER |
| English holdout | 4.73% WER | 4.29% WER | 4.29% WER |
| Japanese calibration | 16.60% CER | 16.01% CER | 9.75% CER |
| Japanese holdout | 22.84% CER | 20.47% CER | 20.47% CER |

The partial `ja-reference-02` reference alone accounts for about six points of the
Japanese calibration aggregate. Only 0–5% of reference speech per case falls where
generation produced no segment. Removing fillers changed Japanese holdout by only
about one point: the remaining disagreement is mostly omitted short words and hedges
(`みたいな`, `とか`, sentence-final `ね`) in conversation, speaker names in
`ja-reference-20`, and kana/kanji choices. `ja-reference-16` output is largely
hiragana (`じかん` for `時間`), a readability defect rather than only a scoring
difference. These remain open generation issues.

A fixed disfluent `initial_prompt` was tried on the calibration split only
(`prompt-trial-results.json`). It increased disagreement on 22 of 28 cases
(English 6.37% → 9.23% WER, Japanese 9.75% → 11.45% CER excluding ja-reference-02)
and dropped whole English sentences. It was rejected; the holdout was not run.
The in-process control reproduced the baseline text exactly.

## Kana-only chunk re-decoding

The `ja-reference-16` hiragana output came from whole ASR chunks decoded in
learner-reader style (spaced hiragana, no kanji). The engine now re-decodes only
such chunks, with an ordinary kanji-mixed prompt, and accepts a re-decode only
when it contains kanji and keeps 50–120% of the original length. The symptom was
found in a holdout case; the rule uses no reference text. ASR-only trial on all
20 Japanese cases (`kana-trial-results.json`):

- 7 chunks matched; 5 in `ja-reference-16` were accepted (e.g. `2じかんぐらい かけて
  りょうり` → `2時間ぐらいかけて、料理`; `おふろに はえる` → `お風呂に入る`).
- Natural kana chunks in `ja-reference-05`/`-07` were kept; the length guard
  refused a re-decode that dropped a repeated `ありがとうございます`.
- `ja-reference-16` CER 18.61% → 13.99%; Japanese holdout 20.47% → 20.00%;
  calibration unchanged (no chunk matched). Some accepted re-decodes omit a
  repeated backchannel (`うん`, a second `わかる`).
- An aligned end-to-end run of `ja-reference-16` kept all 5 re-decodes with
  word timing and recorded them in `meta.orthography_redecodes`.

## Error sources and further trials

English v3 also normalizes curly apostrophes (five references use `’`, which
previously scored every contraction as an error) and `OK`/`okay`.

Japanese scored on kana readings (pykakasi diagnostic, not a gate) drops from
9.75% to 6.72% calibration and 20.00% to 16.55% holdout CER with kana
re-decoding: about three points are kanji/kana/numeral spelling choices. The
remaining holdout error is dominated by deletions in conversational podcasts
(`ja-reference-05`, `-15`, `-18`, `-20`): short words and discourse markers such
as `あの`, `なんか`, `まあ`, `その`, `ね`, `で`.

Shorter VAD merge chunks (`chunk-trial-results.json`, calibration only) made no
material difference: English 5.73% → 5.77% (15 s and 10 s), Japanese 9.75% →
9.60% (15 s) / 9.65% (10 s). Rejected; the production chunk size stays 30 s.

A Japanese-specific model, `kotoba-whisper-v2.0` (CTranslate2), was tried on the
Japanese calibration split (`kotoba-trial-results.json`). Through the production
WhisperX path it scored 30.75% CER (30 s) and 30.52% (15 s) against 9.75% for
large-v3; with its documented faster-whisper settings it was still 14.7–38.2% on
five cases where large-v3 scored 1.6–21.8%. Errors include misrecognized content
(`化学式の読み方はo2だ` → `学学学医している`), not only omissions. It was about 2–3×
faster but was rejected; all languages stay on large-v3, with no language routing.

## Scoring v4: acceptable differences separated

Scoring v4 (`reference-scores-v4.json`, same generated outputs) keeps every
v3 rule and additionally forgives only differences that are not misheard words:
same-reading kana/kanji/numeral spellings in Japanese (1,010 characters across
the corpus), English word-spacing splits/joins (182 words), 343 speaker labels
in `ja-reference-20` via `reference-corrections.json`, and generated text outside
the caption time span. Homophones written with different kanji remain errors.

| Set | v3 all cases | v4 all cases | v4 full-coverage references |
|---|---:|---:|---:|
| English calibration | 5.73% WER | 5.38% WER | 5.38% WER |
| English holdout | 4.29% WER | 4.00% WER | 4.00% WER |
| Japanese calibration | 16.01% CER | 9.49% CER | 7.86% CER |
| Japanese holdout | 20.47% CER | 15.38% CER | 15.38% CER |

With the kana-only chunk re-decoding now in production (ASR-only trial outputs),
Japanese is 7.85% (calibration, full coverage) and 14.88% (holdout).

Sampled accepted differences were spelling-only (`けっこう`/`結構`, `にんじん`/`人参`,
`smart phone`/`smartphone`). The largest remaining differences are mostly real:
dropped conversational stretches, `瑠璃光院` → `ルリ公園`, and an English phrase in
`ja-reference-15` generated as `最高`. Some still need listening review rather
than rules: `ja-reference-06` generated a subscribe outro absent from its captions
(possibly uncaptioned speech), and `en-reference-08` captions contain written
mathematical notation rather than the spoken words.

## Outstanding acceptance work

The generation disagreements still need acoustic adjudication, and incomplete or
suspect references need replacement/review before they can define accuracy gates.
Speaker identity, word timing, sentence/expression boundaries and the spoken-only
requirement remain unverified. Twenty first-pass files contain some unknown word
speakers; 26 contain short rows flagged for review. Reproducibility and parser
success do not waive these requirements. No benchmark output was imported into
the user's application library.
