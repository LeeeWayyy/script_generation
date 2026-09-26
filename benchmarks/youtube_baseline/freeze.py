"""Freeze exact 16-kHz PCM inputs once; never silently replace pinned audio."""
import argparse
import json
import re
import wave
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.run import _sha256
from transcript.audio import extract_audio
from transcript.ingest import _download_url
from .run import save


def verified_audio(case, root):
    path = (root / case['media']).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Pinned audio must exist inside the corpus directory')
    if _sha256(path) != case['media_sha256']:
        raise ValueError('Pinned audio changed: ' + case['id'])
    return path


def freeze(manifest, output):
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads(manifest.read_text(encoding='utf-8'))
    pinned_path = output / 'pinned.json'
    pinned = json.loads(pinned_path.read_text(encoding='utf-8')) if pinned_path.exists() else {
        'schema_version': 1, 'source_manifest_sha256': _sha256(manifest),
        'created_at': datetime.now(timezone.utc).isoformat(), 'cases': [],
    }
    if pinned['source_manifest_sha256'] != _sha256(manifest):
        raise ValueError('Source manifest changed; create a separate corpus version')
    existing = {c['id']: c for c in pinned['cases']}
    for case in source['cases']:
        if case['id'] in existing:
            verified_audio(existing[case['id']], output)
            continue
        if not re.fullmatch(r'[a-z0-9-]+', case['id']):
            raise ValueError('Unsafe case identifier')
        folder = output / case['id']
        folder.mkdir(exist_ok=True)
        media = _download_url(case['url'], folder)
        info = json.loads(media.with_name(media.stem + '.info.json').read_text(encoding='utf-8'))
        if any(k != 'live_chat' and v for k, v in info.get('subtitles', {}).items()):
            raise ValueError('Creator captions appeared since selection: ' + case['id'])
        audio = extract_audio(media, folder)
        with wave.open(str(audio), 'rb') as stream:
            assert (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) == (16000, 1, 2)
            duration = stream.getnframes() / stream.getframerate()
        frozen = {**case, 'source_duration_s': case['duration_s'], 'duration_s': duration,
                  'media': audio.relative_to(output).as_posix(), 'media_sha256': _sha256(audio),
                  'download_sha256': _sha256(media), 'download_format': info.get('format_id'),
                  'frozen_at': datetime.now(timezone.utc).isoformat()}
        pinned['cases'].append(frozen)
        save(pinned_path, pinned)
        print('Pinned', case['id'], round(duration, 3), frozen['media_sha256'], flush=True)
    return pinned_path


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    freeze(a.manifest, a.output)
