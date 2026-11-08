"""Real-GPU integration check. Run against a COPY of the data directory:

    STUDIO_DATA=/path/to/copy python -X utf8 tests/check_experiments_gpu.py

Short bounded runs only (a few optimizer updates). Verifies legacy migration + generation, experiment
isolation, resume, weight changes, BEST/LATEST/snapshot selection, export/import and Chatterbox.
"""
import hashlib
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from studio import storage, registry, datasets, generation, config as settings  # noqa: E402
from studio.backends import kitten  # noqa: E402

assert 'STUDIO_DATA' in os.environ, 'Refusing to run against the live data directory. Set STUDIO_DATA to a copy.'
import soundfile as sf  # noqa: E402
import torch  # noqa: E402


def report(message, **fields):
    print(' ', message, flush=True)


never = lambda: False  # noqa: E731


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(variant, text='Hello. This is a short test of my saved voice.'):
    before = {p.name for p in (storage.DATA / 'outputs').glob('*.json')}
    generation.run(report, never, variants=[generation.validate_variant(variant)], texts=[text])
    new = [storage.read(p) for p in (storage.DATA / 'outputs').glob('*.json') if p.name not in before]
    assert len(new) == 1, new
    audio, rate = sf.read(storage.DATA / 'outputs' / new[0]['file'])
    assert len(audio) > rate * .5 and abs(audio).max() > 1e-3, 'silent or empty audio'
    print(f"PASS generate {variant.get('backend')}/{variant.get('kind')} {variant.get('checkpoint', '')}: "
          f"{len(audio) / rate:.2f}s -> checkpoint={new[0].get('checkpoint')} update={new[0].get('update')}", flush=True)
    return new[0]


registry.migrate()
legacy = next(e for e in registry.listing() if e.get('legacy'))
legacy_hash = digest(registry.checkpoint_path(legacy, 'latest'))
generate({'backend': 'kittentts2-lora', 'kind': 'finetuned', 'experiment_id': legacy['id'], 'checkpoint': 'latest'})
generate({'backend': 'kittentts2-lora', 'kind': 'base', 'voice': 'Bruno', 'seed': 1})

exp = registry.create('GPU smoke', datasets.DEFAULT_ID, 'Speaker 1',
                      {'rank': 4, 'alpha': 8, 'targets': ['q_proj', 'v_proj', 'o_proj'], 'seed': 3})
run = {'epochs': 1, 'max_updates': 4, 'batch_size': 1, 'accumulation': 1, 'lr': 1e-4, 'validate_every': 2,
       'save_every': 2, 'keep_checkpoints': 1, 'warmup_updates': 1, 'scheduler': 'cosine', 'gpus': [0]}
kitten.train(report, never, experiment_id=exp['id'], run=run)
exp = registry.load(exp['id'])
assert exp['updates'] == 4 and exp['best'] and exp['latest'], exp
snapshots = [n for n in exp['checkpoints'] if n.startswith('update-')]
assert len(snapshots) == 1, snapshots  # Retention: keep 1 numbered snapshot.
first = torch.load(registry.checkpoint_path(exp, 'latest'), weights_only=True)
assert all(v.abs().sum() > 0 for k, v in first['adapter'].items() if k.endswith('.b')), 'adapter B stayed zero: no learning'
split = storage.read(registry.root() / exp['id'] / 'split.json')
assert split['kind'] == 'recording'
print('PASS train: 4 updates, best + latest + 1 snapshot, adapter weights changed, recording-level split', split['recordings'], flush=True)

kitten.train(report, never, experiment_id=exp['id'], run=dict(run, max_updates=2))
exp = registry.load(exp['id'])
second = torch.load(registry.checkpoint_path(exp, 'latest'), weights_only=True)
assert exp['updates'] == 6 and second['step'] == 6, exp['updates']
assert any(not torch.equal(first['adapter'][k], second['adapter'][k]) for k in first['adapter'])
assert second['optimizer']['state'][0]['step'] > first['optimizer']['state'][0]['step']
print('PASS resume: 4 -> 6 updates, optimizer state continued, weights changed', flush=True)

try:
    settings.check_resume(exp['config'], settings.validate({**exp['config'], 'rank': 8}))
    raise AssertionError('fixed setting change was accepted')
except ValueError:
    print('PASS fixed settings cannot change on resume', flush=True)
try:
    kitten.validate_checkpoint(second, {'architecture': dict(exp['architecture'], rank=8)})
    raise AssertionError('incompatible architecture loaded')
except ValueError:
    print('PASS incompatible adapter shape refused', flush=True)
assert digest(registry.checkpoint_path(registry.load(legacy['id']), 'latest')) == legacy_hash
print('PASS isolation: legacy checkpoint unchanged', flush=True)

best = generate({'backend': 'kittentts2-lora', 'kind': 'finetuned', 'experiment_id': exp['id'], 'checkpoint': 'best', 'seed': 5})
latest = generate({'backend': 'kittentts2-lora', 'kind': 'finetuned', 'experiment_id': exp['id'], 'checkpoint': 'latest', 'seed': 5})
snap = sorted(n for n in exp['checkpoints'] if n.startswith('update-'))[-1]
specific = generate({'backend': 'kittentts2-lora', 'kind': 'finetuned', 'experiment_id': exp['id'], 'checkpoint': snap, 'seed': 5})
assert best['checkpoint'] == 'best.pt' and latest['checkpoint'] == 'latest.pt' and specific['checkpoint'] == snap
assert latest['update'] == 6 and best['update'] == exp['best']['update']
print('PASS BEST/LATEST/specific checkpoint selection', flush=True)

from studio.export import export, import_archive  # noqa: E402
archive = export(exp['id'], 'best')
with zipfile.ZipFile(archive) as z:
    names = set(z.namelist())
    assert names == {'model.json', 'adapter.safetensors', 'reference.wav', 'README.txt', 'SHA256SUMS.txt', 'infer.py'}, names
    assert z.testzip() is None
imported = import_archive(archive)
loaded = torch.load(registry.checkpoint_path(imported, 'latest'), weights_only=True)
original = torch.load(registry.checkpoint_path(exp, 'best'), weights_only=True)
assert all(torch.equal(loaded['adapter'][k], original['adapter'][k]) for k in original['adapter'])
generate({'backend': 'kittentts2-lora', 'kind': 'finetuned', 'experiment_id': imported['id'], 'checkpoint': 'latest', 'seed': 5})
print(f'PASS export/import round trip ({archive.stat().st_size / 1e6:.1f} MB, adapter tensors identical)', flush=True)

from studio.backends import chatterbox  # noqa: E402
if chatterbox.availability()['available']:
    clip = next(r['id'] for r in (storage.read(p) for p in storage.clips()) if r.get('seconds', 0) >= 6)
    generate({'backend': 'chatterbox', 'reference_clip': clip, 'exaggeration': .5, 'cfg_weight': .5, 'seed': 1})
else:
    print('SKIP Chatterbox:', chatterbox.availability()['reason'])
print('ALL GPU CHECKS PASSED', flush=True)
