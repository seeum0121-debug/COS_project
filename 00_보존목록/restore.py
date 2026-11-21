"""Restore all preserved original paths from the release assets; never overwrite files."""
import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


def safe_path(root, relative):
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()) or candidate == root.resolve():
        raise ValueError(f'Unsafe path: {relative}')
    return candidate


def restore(manifest, receipts, archives, destination=None):
    with manifest.open(encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['status'] == '보존']
    targets = {}
    for row in rows:
        key = (row['archive'], row['member'])
        if row['member'] != row['sha256']:
            raise ValueError('Invalid manifest object hash')
        targets.setdefault(key, []).append(row)
        if destination:
            out = safe_path(destination, row['root'] + '/' + row['path'])
            if out.exists():
                raise FileExistsError(f'Refusing overwrite: {out}')
    seen = set()
    for item in receipts['archives']:
        ap = archives / item['name']
        if ap.stat().st_size != item['bytes']:
            raise ValueError(f'Archive size mismatch: {ap}')
        with ap.open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != item['sha256']:
                raise ValueError(f'Archive hash mismatch: {ap}')
        with zipfile.ZipFile(ap) as z:
            for member in z.infolist():
                key = (ap.name, member.filename)
                if key not in targets or key in seen:
                    raise ValueError(f'Unexpected or duplicate object: {key}')
                with z.open(member) as src, tempfile.TemporaryFile() as temp:
                    h = hashlib.sha256()
                    while chunk := src.read(4 * 1024 * 1024):
                        h.update(chunk)
                        if destination:
                            temp.write(chunk)
                    if h.hexdigest() != member.filename:
                        raise ValueError(f'Object hash mismatch: {key}')
                    for row in targets[key]:
                        if int(row['bytes']) != member.file_size:
                            raise ValueError('Object size mismatch')
                        if destination:
                            out = safe_path(destination, row['root'] + '/' + row['path'])
                            out.parent.mkdir(parents=True, exist_ok=True)
                            temp.seek(0)
                            with out.open('xb') as dst:
                                shutil.copyfileobj(temp, dst)
                            with out.open('rb') as f:
                                if hashlib.file_digest(f, 'sha256').hexdigest() != row['sha256']:
                                    raise ValueError(f'Restored file mismatch: {out}')
                seen.add(key)
        print('Verified:', ap.name, flush=True)
    if seen != set(targets):
        raise ValueError('Missing preserved objects')
    return len(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    home = Path(__file__).resolve().parent
    receipts = json.loads((home / '보존검증.json').read_text(encoding='utf-8'))
    count = restore(home / '전체파일목록.csv', receipts, args.archives, args.destination)
    print(f'{count} preserved paths verified; ' + ('restored.' if args.destination else 'verification only.'))
