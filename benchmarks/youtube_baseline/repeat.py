"""Three independent inference passes over identical, hash-verified audio."""
import argparse
import json
from pathlib import Path
from .run import digest, run, save


def semantic_result(data):
    # IDs and server paths identify executions, not generated caption content.
    return {'segments': data['segments'], 'language': data['language'],
            'meta': {k: v for k, v in data.get('meta', {}).items()
                     if k not in ('job_id', 'source')}}


def compare_runs(manifest, output, count):
    cases = json.loads(manifest.read_text(encoding='utf-8'))['cases']
    rows = []
    for case in cases:
        hashes, statuses, errors = [], [], []
        for index in range(1, count + 1):
            root = output / f'run-{index}'
            path = root / (case['id'] + '.json')
            if not path.exists():
                statuses.append('pending')
                continue
            row = json.loads(path.read_text(encoding='utf-8'))
            statuses.append(row.get('status', 'pending'))
            if row.get('status') == 'done':
                payload = json.loads((root / (case['id'] + '.readable.json')).read_text(encoding='utf-8'))
                hashes.append(digest(semantic_result(payload)))
            else:
                errors.append(row.get('error'))
        stable = (len(hashes) == count and len(set(hashes)) == 1)
        repeated_error = (len(errors) == count and len(set(errors)) == 1 and all(s == 'error' for s in statuses))
        rows.append({'case': case['id'], 'media_sha256': case['media_sha256'],
                     'statuses': statuses, 'content_hashes': hashes,
                     'exact_caption_repeatability_pass': stable,
                     'repeatable_error': repeated_error, 'errors': errors})
    result = {'runs': count, 'cases': len(cases),
              'exact_caption_repeatability_passed': sum(r['exact_caption_repeatability_pass'] for r in rows),
              'repeatable_errors': sum(r['repeatable_error'] for r in rows), 'rows': rows}
    save(output / 'repeatability.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--server', default='http://127.0.0.1:8000')
    a = p.parse_args()
    if a.runs < 2:
        p.error('At least two independent runs are required')
    manifest = json.loads(a.manifest.read_text(encoding='utf-8'))
    if not manifest['cases'] or any('media_sha256' not in c for c in manifest['cases']):
        p.error('Freeze the exact media before testing repeatability')
    for i in range(1, a.runs + 1):
        run(a.manifest, a.output / f'run-{i}', a.server)
        result = compare_runs(a.manifest, a.output, a.runs)
        print('Completed pass', i, 'exact-caption matches', result['exact_caption_repeatability_passed'], flush=True)
