"""Timestamped, verified backups written outside the data directory.

    python -m studio.backup models    # checkpoints + everything needed to use them
    python -m studio.backup dataset   # recordings, clips, subtitles, feature caches
    python -m studio.backup all

Archives go to STUDIO_BACKUPS (default: ~/voice-studio-backups). Each archive gets a
.sha256 file and is re-read after writing to verify every member.
"""
import argparse
import hashlib
import os
import time
import zipfile
from pathlib import Path

from .storage import DATA

# Already-compressed or large binary files are stored, not deflated (faster, same size).
STORED = {'.wav', '.mp3', '.pt', '.safetensors', '.zip', '.flac', '.ogg'}


def destination():
    target = Path(os.environ.get('STUDIO_BACKUPS', str(Path.home() / 'voice-studio-backups'))).resolve()
    if target.is_relative_to(DATA):
        raise ValueError('Backups must be written outside the data directory they archive.')
    target.mkdir(parents=True, exist_ok=True)
    return target


def model_files():
    """Weights, reference audio, metadata, subtitle links, splits, configs and logs."""
    patterns = ['models/**/*', 'experiments/**/*', 'subtitles/*', 'jobs/*.json', 'jobs/*.log',
                'samples/*.json', 'clips/*.json', 'datasets.json']
    return patterns


def dataset_files():
    return ['samples/*', 'clips/*', 'features/*', 'subtitles/*', 'converted/*', 'datasets.json']


def collect(patterns):
    files = []
    for pattern in patterns:
        files += [p for p in DATA.glob(pattern) if p.is_file() and not p.name.endswith('.tmp')]
    return sorted(set(files))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def create(kind, patterns):
    files = collect(patterns)
    if not files:
        raise ValueError(f'Nothing to back up for {kind}.')
    stamp = time.strftime('%Y%m%d-%H%M%S')
    target = destination() / f'voice-{kind}-{stamp}.zip'
    partial = target.with_suffix('.partial')
    sums = {}
    with zipfile.ZipFile(partial, 'w', allowZip64=True) as archive:
        for path in files:
            name = path.relative_to(DATA).as_posix()
            sums[name] = sha256(path)
            method = zipfile.ZIP_STORED if path.suffix.lower() in STORED else zipfile.ZIP_DEFLATED
            archive.write(path, name, compress_type=method)
        manifest = '\n'.join(f'{digest}  {name}' for name, digest in sums.items()) + '\n'
        archive.writestr('BACKUP-SHA256SUMS.txt', manifest)
        archive.writestr('BACKUP-README.txt', (
            f'Voice Studio {kind} backup created {stamp} from {DATA}.\n'
            'Restore: stop the server, then extract into the data directory (default: <checkout>/data).\n'
            'Do not overwrite newer files without checking. BACKUP-SHA256SUMS.txt lists per-file checksums.\n'))
    verify(partial, sums)
    os.replace(partial, target)
    digest = sha256(target)
    target.with_name(target.name + '.sha256').write_bytes(f'{digest}  {target.name}\n'.encode())
    size = target.stat().st_size
    print(f'{kind}: {target}  ({len(files)} files, {size / 1e6:.1f} MB, sha256 {digest[:16]}…)', flush=True)
    return target


def verify(path, expected):
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist()) - {'BACKUP-SHA256SUMS.txt', 'BACKUP-README.txt'}
        if names != set(expected):
            raise RuntimeError('Archive contents do not match the source file list.')
        for name, digest in expected.items():
            value = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1 << 20), b''):
                    value.update(block)
            if value.hexdigest() != digest:
                raise RuntimeError(f'Checksum mismatch in archive: {name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['models', 'dataset', 'all'], nargs='?', default='models')
    kind = parser.parse_args().kind
    if kind in ('models', 'all'):
        create('models', model_files())
    if kind in ('dataset', 'all'):
        create('dataset', dataset_files())


if __name__ == '__main__':
    main()
