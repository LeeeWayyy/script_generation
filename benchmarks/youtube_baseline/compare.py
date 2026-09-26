"""Compare the current renderer against frozen first-run raw/readable evidence."""
import argparse
import json
from pathlib import Path
from transcript.readable import readable_transcript
from transcript.types import Transcript, Segment, Word
from .run import assess, digest, save


def compare(manifest, baseline, output):
    output.mkdir(parents=True, exist_ok=True)
    changes = []
    for case in json.loads(manifest.read_text(encoding='utf-8'))['cases']:
        path = baseline / (case['id'] + '.raw.json')
        if not path.exists():
            continue
        raw = json.loads(path.read_text(encoding='utf-8'))
        old = json.loads((baseline / (case['id'] + '.readable.json')).read_text(encoding='utf-8'))
        source = Transcript([Segment(**{**s, 'words': [Word(**w) for w in s['words']]})
                             for s in raw['segments']], raw['language'], raw['meta'])
        new = readable_transcript(source).to_dict()
        assessment = assess(raw, new, case)
        assert 'readable_changed_word_data' not in assessment['errors'], case['id']
        assert 'readable_changed_text' not in assessment['errors'], case['id']
        save(output / (case['id'] + '.readable.json'), new)
        changes.append({'case': case['id'], 'before_rows': len(old['segments']),
                        'after_rows': len(new['segments']), 'changed': old != new,
                        'before_sha256': digest(old), 'after_sha256': digest(new),
                        'assessment': assessment})
    save(output / 'comparison.json', changes)
    print('Compared', len(changes), 'cases;', sum(c['changed'] for c in changes),
          'changed; original speech and every aligned entry preserved')
    return changes


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('baseline', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    compare(a.manifest, a.baseline, a.output)
