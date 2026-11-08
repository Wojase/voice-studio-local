"""Recoverable deletion, restricted to this application's data directory."""
import re
import uuid
from pathlib import Path
from .storage import DATA, read, write


def safe_key(key):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', key or '') or key in ('.', '..'):
        raise ValueError('Invalid item identifier.')
    return key


def collect(kind, key=''):
    if kind in ('samples', 'sample_clips', 'clips', 'outputs', 'converted'):
        safe_key(key)
    if kind == 'samples':
        paths = list((DATA / 'samples').glob(key + '.*'))
        paths += list((DATA / 'subtitles').glob(key + '.*'))
        paths += list((DATA / 'clips').glob(key + '_*'))
        paths += list((DATA / 'features').glob(key + '_*'))
    elif kind == 'sample_clips':
        paths = list((DATA / 'clips').glob(key + '_*')) + list((DATA / 'features').glob(key + '_*'))
        paths += list((DATA / 'subtitles').glob(key + '.*'))
    elif kind == 'clips':
        paths = [DATA / 'clips' / (key + suffix) for suffix in ('.wav', '.json')]
        paths += [DATA / 'features' / (key + '.pt')]
    elif kind in ('outputs', 'converted'):
        paths = [DATA / kind / key]
    elif kind == 'model':
        paths = list((DATA / 'models').iterdir())
    elif kind in ('dataset', 'all'):
        folders = ['samples', 'clips', 'features', 'subtitles']
        if kind == 'all':
            folders += ['models', 'outputs', 'converted']
        paths = [p for name in folders for p in (DATA / name).glob('*')]
        if kind == 'all':
            paths += list((DATA / 'experiments').rglob('*'))
    else:
        raise ValueError('Unknown deletion category.')
    return [p for p in paths if p.is_file()]


def trash(paths, label):
    paths = list(dict.fromkeys(paths))
    for p in paths:
        if not p.resolve().is_relative_to(DATA.resolve()):
            raise ValueError('Path escapes the data directory.')
    if not paths:
        raise ValueError('No matching files.')
    key = uuid.uuid4().hex
    folder = DATA / 'trash' / key
    folder.mkdir(parents=True)
    manifest = {'id': key, 'label': label, 'paths': [str(p.relative_to(DATA)) for p in paths]}
    write(folder / 'manifest.json', manifest)
    moved = []
    try:
        for source in paths:
            target = folder / source.relative_to(DATA)
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            moved.append((source, target))
    except Exception:
        for source, target in reversed(moved):
            target.rename(source)
        raise
    return manifest


def restore(key):
    safe_key(key)
    folder = DATA / 'trash' / key
    manifest = read(folder / 'manifest.json')
    if not manifest:
        raise ValueError('Trash entry not found.')
    pairs = [(folder / p, DATA / p) for p in manifest['paths']]
    for source, target in pairs:
        if not source.resolve().is_relative_to(folder.resolve()) or not target.resolve().is_relative_to(DATA.resolve()):
            raise ValueError('Invalid restore path.')
        if target.exists():
            raise ValueError('Restore would overwrite newer files. Remove the conflicting items first.')
        if not source.is_file():
            raise ValueError('Trash entry is incomplete.')
    moved = []
    try:
        for source, target in pairs:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            moved.append((source, target))
    except Exception:
        for source, target in reversed(moved):
            target.rename(source)
        raise
    (folder / 'manifest.json').unlink()
    return manifest


def purge(key):
    safe_key(key)
    folder = (DATA / 'trash' / key).resolve()
    if folder.parent != (DATA / 'trash').resolve() or not (folder / 'manifest.json').is_file():
        raise ValueError('Trash entry not found.')
    entries = list(folder.rglob('*'))
    if any(not p.resolve().is_relative_to(folder) for p in entries):
        raise ValueError('Trash contains an unsafe path.')
    for p in entries:
        if p.is_file():
            p.unlink()
    for p in sorted((p for p in entries if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        p.rmdir()
    folder.rmdir()
