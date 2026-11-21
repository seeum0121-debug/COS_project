"""Record distributed files after an intentional package build or update."""
from pathlib import Path
from common import sha256, write_json

ROOT = Path(__file__).resolve().parent


def main():
    files = {}
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or '.venv' in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative in ('checksums.json', 'verification.json') or path.suffix == '.log':
            continue
        files[relative] = sha256(path)
    write_json(ROOT/'checksums.json', {'algorithm': 'sha256', 'files': files})
    print('Sealed', len(files), 'files')


if __name__ == '__main__':
    main()
