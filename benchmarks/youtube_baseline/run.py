"""Run the no-creator-captions corpus through the real server, retaining evidence.

This measures pipeline correctness and diagnostic quality signals, not WER/CER.
Those require independently reviewed references (see benchmarks/run.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def assess(raw, readable, case):
    errors, warnings = [], []
    segments = readable.get('segments', [])
    raw_words = [w for s in raw.get('segments', []) for w in s.get('words', [])]
    words = [w for s in segments for w in s.get('words', [])]
    if not segments or not any(s.get('text', '').strip() for s in segments):
        errors.append('empty_transcript')
    if words != raw_words:
        errors.append('readable_changed_word_data')
    def text(rows):
        return re.sub(r'\s+', '', ''.join(s.get('text', '') for s in rows))
    if text(segments) != text(raw.get('segments', [])):
        errors.append('readable_changed_text')
    meta = raw.get('meta', {})
    if meta.get('transcript_source') == 'youtube_manual_captions' or meta.get('caption_language'):
        errors.append('creator_captions_used')
    for key in ('align_succeeded', 'diarize_succeeded'):
        if meta.get(key) is not True:
            errors.append(key + '_not_true')
    if raw.get('language') != case['language']:
        errors.append('unexpected_language')
    fallbacks = readable.get('meta', {}).get('readable', {}).get('fallbacks', [])
    indices = [f.get('segment_index') for f in fallbacks]
    if any(type(i) is not int or not 0 <= i < len(segments) for i in indices):
        errors.append('invalid_fallback_index')
    if len(indices) != len(set(indices)):
        errors.append('duplicate_fallback_index')
    def valid_bounds(start, end):
        if start is None and end is None:
            return True
        return (all(type(t) in (int, float) and math.isfinite(t) for t in (start, end))
                and 0 <= start <= end <= 864000)
    if any(not valid_bounds(f.get('source_start'), f.get('source_end')) for f in fallbacks):
        errors.append('invalid_fallback_timing')
    if any(not valid_bounds(s.get('start'), s.get('end')) for s in segments):
        errors.append('invalid_segment_bounds')
    starts = [s['start'] for s in segments if type(s.get('start')) in (int, float)]
    if any(b < a for a, b in zip(starts, starts[1:])):
        errors.append('nonmonotonic_segment_starts')
    if len(json.dumps(readable).encode()) > 32000000 or len(segments) > 100000:
        errors.append('app_payload_limit_exceeded')
    if any(not isinstance(s.get('text'), str) or len(s['text']) > 50000 for s in segments):
        errors.append('invalid_segment_text')
    invalid, missing = 0, 0
    timed_words = 0
    for kind, items in (('segment', segments), ('word', words)):
        for item in items:
            start, end = item.get('start'), item.get('end')
            if start is None or end is None:
                if kind == 'word':
                    missing += 1
                continue
            if (any(type(t) not in (int, float) or not math.isfinite(t) for t in (start, end))
                    or start < 0 or end < start or end > case['duration_s'] + 2):
                invalid += 1
            elif kind == 'word':
                timed_words += 1
    if invalid:
        errors.append('invalid_or_out_of_media_timing')
    if missing:
        warnings.append('missing_word_timing')
    short = [i for i, s in enumerate(segments)
             if len(re.sub(r'\W', '', s.get('text', ''))) <= 2
             and s.get('text', '').strip()]
    if short:
        warnings.append('short_rows_require_review')
    repeated = [i for i in range(2, len(segments))
                if segments[i]['text'] == segments[i-1]['text'] == segments[i-2]['text']]
    if repeated:
        warnings.append('repeated_text_requires_review')
    uncertain = sum(w.get('speaker') is None for w in words)
    if uncertain:
        warnings.append('unknown_word_speakers')
    total_text = text(segments)
    ja_chars = sum('\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff' for c in total_text)
    if case['language'] == 'ja' and total_text and ja_chars / len(total_text) < .25:
        warnings.append('low_japanese_script_share')
    ends = [s['end'] for s in segments if type(s.get('end')) in (int, float)]
    end_fraction = max(ends, default=0) / case['duration_s']
    if end_fraction < .8:
        warnings.append('early_transcript_end_requires_listening')
    return {
        'functional_pass': not errors, 'errors': errors, 'quality_warnings': warnings,
        'raw_rows': len(raw.get('segments', [])), 'readable_rows': len(segments),
        'aligned_entries': len(words), 'timed_entry_fraction': timed_words / max(len(words), 1),
        'unknown_speaker_fraction': uncertain / max(len(words), 1),
        'fallback_count': len(fallbacks), 'short_row_indices': short,
        'repeated_row_indices': repeated, 'last_timestamp_fraction': end_fraction,
        'independent_reference': None, 'wer': None, 'cer': None,
        'accuracy_status': 'not_scored_without_reviewed_reference',
        'acceptance_pass': False,
        'unresolved_requirements': ['independent_word_accuracy_review',
                                    'acoustic_timing_and_speaker_review',
                                    'sentence_and_expression_boundary_review'],
    }


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def run(manifest_path, output, server):
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    session = requests.Session()
    token = os.environ.get('TRANSCRIPT_TOKEN')
    if not token:
        raise ValueError('TRANSCRIPT_TOKEN must be set; never print it')
    session.headers['Authorization'] = 'Bearer ' + token
    base = server.rstrip('/')

    def get(path, **kwargs):
        response = session.get(base + path, timeout=30, **kwargs)
        response.raise_for_status()
        return response.json()

    health = get('/health')
    try:
        commit = subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[2]), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    save(output / ('environment-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json'), {
        'created_at': datetime.now(timezone.utc).isoformat(), 'server': base,
        'health': health, 'runner_platform': platform.platform(), 'checkout_commit': commit,
        'manifest_sha256': digest(manifest), 'configuration': {'align': True, 'diarize': True},
    })
    rows = []
    for case in manifest['cases']:
        path = output / (case['id'] + '.json')
        row = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'case': case}
        if row.get('status') in ('done', 'error'):
            if row['status'] == 'done':
                raw = json.loads((output / (case['id'] + '.raw.json')).read_text(encoding='utf-8'))
                readable = json.loads((output / (case['id'] + '.readable.json')).read_text(encoding='utf-8'))
                row['assessment'] = assess(raw, readable, case)
                save(path, row)
            rows.append(row)
            continue
        started = time.monotonic()
        try:
            if not row.get('job_id'):
                # Preserve the real service configuration and queue; never restart
                # it or launch a second GPU model for benchmarking.
                while get('/health')['queued_or_running']:
                    time.sleep(5)
                response = session.post(base + '/jobs', data={
                    'url': case['url'], 'language': case['language'],
                    'align': 'true', 'diarize': 'true',
                }, timeout=60)
                response.raise_for_status()
                row['job_id'] = response.json()['id']
                row['submitted_at'] = datetime.now(timezone.utc).isoformat()
                save(path, row)
            deadline = time.monotonic() + 7200
            while True:
                status = get('/jobs/' + row['job_id'])
                if status['status'] in ('done', 'error'):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError('Job still pending after two hours; retain job_id for diagnosis')
                time.sleep(3)
            row['server_status'] = status
            if status['status'] == 'error':
                raise RuntimeError(status.get('error', 'server error'))
            raw = get('/jobs/' + row['job_id'] + '/result', params={'format': 'json'})
            readable = get('/jobs/' + row['job_id'] + '/result', params={'format': 'json', 'readable': 'true'})
            save(output / (case['id'] + '.raw.json'), raw)
            save(output / (case['id'] + '.readable.json'), readable)
            row.update(status='done', assessment=assess(raw, readable, case),
                       raw_sha256=digest(raw), readable_sha256=digest(readable))
            row['server_elapsed_s'] = status['finished_at'] - status['started_at']
            row['realtime_factor'] = row['server_elapsed_s'] / case['duration_s']
        except Exception as exc:
            row.update(status='error', error=str(exc))
        row['client_elapsed_s'] = time.monotonic() - started
        save(path, row)
        rows.append(row)
        save(output / 'summary.json', {
            'cases': len(manifest['cases']), 'completed': len(rows),
            'functional_passed': sum(r.get('assessment', {}).get('functional_pass', False) for r in rows),
            'acceptance_passed': sum(r.get('assessment', {}).get('acceptance_pass', False) for r in rows),
            'rows': rows,
        })
        print(case['id'], row['status'], row.get('assessment', {}).get('errors', []),
              row.get('error', ''), flush=True)
    save(output / 'summary.json', {
        'cases': len(manifest['cases']), 'completed': len(rows),
        'functional_passed': sum(r.get('assessment', {}).get('functional_pass', False) for r in rows),
        'acceptance_passed': sum(r.get('assessment', {}).get('acceptance_pass', False) for r in rows),
        'rows': rows,
    })
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--server', default='http://127.0.0.1:8000')
    args = parser.parse_args()
    result = run(args.manifest, args.output, args.server)
    raise SystemExit(0 if all(r.get('assessment', {}).get('functional_pass') for r in result) else 1)
