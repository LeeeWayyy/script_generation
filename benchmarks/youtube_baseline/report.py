"""Reassess saved evidence without rerunning inference or altering baseline JSON."""
import argparse
import json
from collections import Counter
from pathlib import Path

try:
    from .run import assess, digest, save
except ImportError:
    from run import assess, digest, save


def report(manifest_path, results):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    rows, review = [], []
    for case in manifest['cases']:
        path = results / (case['id'] + '.json')
        row = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'case': case, 'status': 'pending'}
        raw_path = results / (case['id'] + '.raw.json')
        readable_path = results / (case['id'] + '.readable.json')
        if raw_path.exists() and readable_path.exists():
            raw = json.loads(raw_path.read_text(encoding='utf-8'))
            readable = json.loads(readable_path.read_text(encoding='utf-8'))
            row['assessment'] = assess(raw, readable, case)
            segments = readable['segments']
            flagged = set(row['assessment']['short_row_indices'] + row['assessment']['repeated_row_indices'])
            for i, segment in enumerate(segments):
                reasons = []
                if i in flagged:
                    reasons.append('short_or_repeated_row')
                if segment.get('start') is None or segment.get('end') is None:
                    reasons.append('missing_row_timing')
                if segment.get('speaker') is None:
                    reasons.append('uncertain_speaker')
                if any(w.get('start') is None or w.get('end') is None for w in segment.get('words', [])):
                    reasons.append('missing_word_timing')
                if i and segment.get('speaker') == segments[i-1].get('speaker') and segment.get('speaker'):
                    left = segments[i-1]
                    if not left['text'].endswith(('.', '?', '!', '。', '？', '！')) and len(segment.get('words', [])) <= 3:
                        reasons.append('possible_stranded_sentence_tail')
                if reasons:
                    review.append({'case': case['id'], 'segment_index': i, 'start': segment.get('start'),
                                   'end': segment.get('end'), 'text': segment['text'],
                                   'previous_text': segments[i-1]['text'] if i else None,
                                   'reasons': reasons, 'status': 'unreviewed',
                                   'audio_url': case['url'] + '&t=' + str(max(0, int(segment.get('start') or 0) - 2))})
            row['review_coverage'] = {'total_rows': len(segments), 'independently_reviewed_rows': 0,
                                      'readable_sha256': digest(readable)}
        rows.append(row)
    output = {
        'manifest_sha256': digest(manifest), 'cases': len(rows),
        'languages': dict(Counter(r['case']['language'] for r in rows)),
        'input_hours': sum(r['case']['duration_s'] for r in rows) / 3600,
        'completed': sum(r.get('status') in ('done', 'error') for r in rows),
        'functional_passed': sum(r.get('assessment', {}).get('functional_pass', False) for r in rows),
        'acceptance_passed': 0,
        'acceptance_status': 'NOT_ACCEPTED: unresolved functional failures and/or independent review requirements',
        'review_flags': len(review), 'rows': rows,
    }
    save(results / 'assessment.json', output)
    save(results / 'review-queue.json', review)
    lines = ['# YouTube baseline — actual Windows results', '',
             '**Full acceptance: NOT PASSED.** No unresolved case or caption is waived.', '',
             f"{output['completed']}/{len(rows)} inputs completed; {output['functional_passed']} pass executable functional gates.",
             f"{output['input_hours']:.2f} input hours; {len(review)} flagged rows need examination. All rows still require independent accuracy review.",
             '', '| Case | Video | Minutes | Job | Rows | Functional result |',
             '|---|---|---:|---|---:|---|']
    for row in rows:
        c, a = row['case'], row.get('assessment', {})
        state = ('PASS' if a.get('functional_pass') else ', '.join(a.get('errors', []))) or row.get('status', 'pending')
        lines.append(f"| {c['id']} | [{c['title'].replace('|', '/')} ]({c['url']}) | {c['duration_s']/60:.1f} | {row.get('job_id', '—')} | {a.get('readable_rows', '—')} | {state} |")
    lines += ['', '## Interpretation', '',
              'Functional gates cover inference flags, metadata preservation and structural app invariants. They do not establish acoustic accuracy.',
              'No independently reviewed reference exists yet, so WER/CER and zero-error claims are withheld. The original outputs, failures, hashes and per-row review queue are retained.',
              'The queue prioritizes suspected issues; unflagged rows are not implicitly approved. Topic/type selection and spoken-language eligibility also require verification against content.',
              'No generated speech was rewritten to fit expectations, and no failed video was removed from the corpus.']
    (results / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('results', type=Path)
    a = p.parse_args()
    result = report(a.manifest, a.results)
    print({k: v for k, v in result.items() if k != 'rows'})
