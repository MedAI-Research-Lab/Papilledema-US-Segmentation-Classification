"""Verify the public source/data package against its SHA-256 inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'provenance/release_manifest.json'

def verify(root: Path = ROOT) -> dict:
    manifest = json.loads((root / 'provenance/release_manifest.json').read_text(encoding='utf-8'))
    failures = []
    for record in manifest['files']:
        relative = Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts:
            failures.append({'path': record['path'], 'reason': 'unsafe inventory path'})
            continue
        path = root / relative
        if not path.is_file():
            failures.append({'path': record['path'], 'reason': 'file not found'})
            continue
        # Source text may be normalized by Git. Compare the canonical LF bytes
        # for text, and exact bytes for binary content.
        payload = path.read_bytes()
        if record.get('text_lf'):
            payload = payload.replace(b'\r\n', b'\n')
        if hashlib.sha256(payload).hexdigest() != record['sha256']:
            failures.append({'path': record['path'], 'reason': 'content hash differs'})
    result = {'checked_files': len(manifest['files']), 'valid': not failures, 'failures': failures}
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    verify(parser.parse_args().root.resolve())
