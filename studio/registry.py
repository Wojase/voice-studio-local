"""Persistent experiment registry. Each experiment owns its folder:

experiments/<id>/experiment.json   identity, base model, architecture, config, status, checkpoint index
experiments/<id>/checkpoints/      latest.pt, best.pt, update-000025.pt ...
experiments/<id>/metrics.jsonl     training and validation loss per update
experiments/<id>/train.log         readable log of every run
experiments/<id>/split.json        persisted train/validation membership
experiments/<id>/reference.wav     reference clip used for decoding conditioning
"""
import hashlib
import platform
import re
import shutil
import uuid
from pathlib import Path

from . import storage, config as settings
from .storage import read, write, now

KITTEN = 'kittentts2-lora'
SNAPSHOT = re.compile(r'update-\d{6,}\.pt')


def root():
    return storage.DATA / 'experiments'


def folder(experiment_id):
    if not re.fullmatch(r'[a-f0-9]{12}', experiment_id or ''):
        raise ValueError('Invalid experiment identifier.')
    path = root() / experiment_id
    if not (path / 'experiment.json').is_file():
        raise ValueError('Experiment not found.')
    return path


def load(experiment_id):
    return read(folder(experiment_id) / 'experiment.json')


def save(experiment):
    experiment['updated'] = now()
    path = root() / experiment['id']
    path.mkdir(parents=True, exist_ok=True)
    (path / 'checkpoints').mkdir(exist_ok=True)
    write(path / 'experiment.json', experiment)
    return experiment


def listing(include_archived=True):
    rows = [read(p) for p in root().glob('*/experiment.json')]
    rows = [r for r in rows if r and (include_archived or not r.get('archived'))]
    return sorted(rows, key=lambda r: r.get('created', ''), reverse=True)


def software():
    import importlib.metadata as metadata
    versions = {'python': platform.python_version()}
    for name in ('torch', 'kittenml', 'transformers', 'safetensors', 'huggingface_hub'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def base_identity():
    from .engine import BASE, REVISION
    return {'id': BASE, 'revision': REVISION, 'weights': 'emb4'}


def create(name, dataset_id, speaker, values, backend=KITTEN, **extra):
    from . import datasets
    dataset = datasets.find(dataset_id)
    if speaker not in dataset['speakers']:
        raise ValueError('Choose a speaker belonging to this dataset.')
    if backend != KITTEN:
        raise ValueError('Only the KittenTTS2 LoRA backend supports training experiments.')
    values = dict(values or {})
    # Defaults follow the server profile (rental: 2 GPUs, batch 4 per GPU; local: 1 GPU, batch 1).
    import os
    values.setdefault('gpus', list(range(max(1, int(os.environ.get('STUDIO_GPUS', '1'))))))
    values.setdefault('batch_size', int(os.environ.get('STUDIO_BATCH_SIZE', '1')))
    cfg = settings.validate(values)
    experiment = {
        'id': uuid.uuid4().hex[:12], 'name': datasets.clean_name(name), 'backend': backend,
        'base': base_identity(), 'architecture': settings.architecture(cfg),
        'dataset_id': dataset_id, 'speaker': speaker, 'config': cfg, 'seed': cfg['seed'],
        'created': now(), 'software': software(), 'status': 'new', 'archived': False,
        'updates': 0, 'epochs': 0, 'checkpoints': {}, 'best': None, 'latest': None,
        'runs': [], 'resumable': True, 'legacy': False, 'imported': False}
    experiment.update(extra)
    return save(experiment)


def duplicate(experiment_id, name=None):
    source = load(experiment_id)
    # Duplicates copy settings only: they start from fresh adapter weights.
    return create(name or (source['name'] + ' (copy)')[:80], source['dataset_id'], source['speaker'],
                  {k: v for k, v in source['config'].items() if k in settings.FIXED or k in settings.RESUMABLE},
                  duplicated_from=experiment_id)


def rename(experiment_id, name):
    from .datasets import clean_name
    experiment = load(experiment_id)
    experiment['name'] = clean_name(name)
    return save(experiment)


def set_archived(experiment_id, archived):
    experiment = load(experiment_id)
    experiment['archived'] = bool(archived)
    return save(experiment)


def delete(experiment_id, confirm):
    """Moves the experiment folder to recoverable Trash. Requires typing the experiment name."""
    from . import management
    experiment = load(experiment_id)
    if confirm != experiment['name']:
        raise ValueError('Type the exact experiment name to delete it.')
    paths = [p for p in folder(experiment_id).rglob('*') if p.is_file()]
    manifest = management.trash(paths, f"experiment: {experiment['name']}")
    shutil.rmtree(root() / experiment_id, ignore_errors=True)  # Only empty directories remain.
    return manifest


def checkpoint_path(experiment, which='best'):
    """Resolve BEST, LATEST or a named snapshot to a file inside this experiment only."""
    names = experiment.get('checkpoints', {})
    if which in ('best', 'latest'):
        name = (experiment.get(which) or {}).get('name')
        if not name:
            if which == 'best':
                raise ValueError('This experiment has no best checkpoint yet (validation has not run). Choose LATEST.')
            raise ValueError('This experiment has no saved checkpoint yet. Train it first.')
    else:
        name = which
    if name not in names or not (SNAPSHOT.fullmatch(name) or name in ('latest.pt', 'best.pt')):
        raise ValueError('Checkpoint not found in this experiment.')
    path = root() / experiment['id'] / 'checkpoints' / name
    if not path.is_file():
        raise ValueError('Checkpoint file is missing.')
    return path


def record_checkpoint(experiment, name, info):
    experiment.setdefault('checkpoints', {})[name] = info


def prune(experiment, keep):
    """Keep the newest `keep` numbered snapshots; latest.pt and best.pt are never removed."""
    snapshots = sorted((n for n in experiment['checkpoints'] if SNAPSHOT.fullmatch(n)),
                       key=lambda n: experiment['checkpoints'][n]['update'])
    path = root() / experiment['id'] / 'checkpoints'
    for name in snapshots[:max(0, len(snapshots) - keep)]:
        (path / name).unlink(missing_ok=True)
        experiment['checkpoints'].pop(name)


def log(experiment_id, message):
    with (root() / experiment_id / 'train.log').open('a', encoding='utf-8') as stream:
        stream.write(f'{now()}  {message}\n')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def migrate_legacy():
    """Import data/models/latest.pt (+ best.pt, reference, split) as a legacy experiment.

    Files are COPIED; data/models is left untouched. Idempotent: keyed by the legacy file hash.
    """
    import torch
    legacy = storage.DATA / 'models'
    source = legacy / 'latest.pt'
    if not source.is_file():
        return None
    digest = sha256(source)
    marker = storage.DATA / 'migrations.json'
    done = read(marker, {})
    # Recorded once, so deleting the imported experiment later does not re-import it on restart.
    if digest in done.get('legacy_models', {}):
        return None
    saved = torch.load(source, map_location='cpu', weights_only=True)
    from .engine import BASE, REVISION
    if saved.get('base') != BASE or saved.get('revision') != REVISION:
        raise ValueError('Legacy checkpoint uses an unknown base model; it was not imported.')
    architecture = saved.get('architecture') or dict(settings.LEGACY_ARCHITECTURE)
    cfg = settings.validate({**architecture, 'validation_fraction': .1})
    from . import datasets
    datasets.load()
    experiment = create('Legacy model (imported)', datasets.DEFAULT_ID, datasets.find(datasets.DEFAULT_ID)['speakers'][0],
                        cfg, legacy=True, migrated_from={'path': 'models/latest.pt', 'sha256': digest, 'at': now()})
    experiment['software'] = dict(software(), note='Trained before the experiment registry existed; exact versions unknown.')
    target = root() / experiment['id']
    for name in ('latest.pt', 'best.pt'):
        if (legacy / name).is_file():
            shutil.copy2(legacy / name, target / 'checkpoints' / name)
            value = saved if name == 'latest.pt' else torch.load(legacy / name, map_location='cpu', weights_only=True)
            record_checkpoint(experiment, name, {'update': value['step'], 'epochs': value['epochs'],
                                                 'validation_loss': value.get('best_loss') if name == 'best.pt' else None,
                                                 'saved': now(), 'source': 'legacy'})
            experiment['best' if name == 'best.pt' else 'latest'] = {'name': name, 'update': value['step']}
    if (legacy / 'reference.wav').is_file():
        shutil.copy2(legacy / 'reference.wav', target / 'reference.wav')
    split = read(legacy / 'split.json', {})
    # The legacy split is clip-level: adjacent clips of one recording may be in both partitions.
    write(target / 'split.json', {'kind': 'clip-legacy', 'clips': split,
                                  'note': 'Preserved legacy clip-level split. Validation clips can share recordings with training clips.'})
    best = saved.get('best_loss')
    experiment.update(updates=saved['step'], epochs=saved['epochs'], status='ready',
                      best_validation_loss=best if best is not None and best != float('inf') else None)
    save(experiment)
    done.setdefault('legacy_models', {})[digest] = {'experiment_id': experiment['id'], 'at': now()}
    write(marker, done)
    log(experiment['id'], f'Imported legacy checkpoint models/latest.pt (sha256 {digest[:16]}, {saved["step"]} updates).')
    return experiment


def migrate():
    from . import datasets
    datasets.load()
    datasets.migrate()
    try:
        migrate_legacy()
    except Exception as exc:  # Never block the panel; the legacy files stay where they are.
        print('Legacy model migration skipped:', exc, flush=True)
