"""CPU tests for the experiment registry, settings, split, checkpoint selection, migration and archives."""
import hashlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from torch import nn

from studio import storage, registry, datasets, config as settings, management, export, locking  # noqa: F401  (bind locking.DATA before patching)
from studio.backends import kitten
from studio.engine import adapters, BASE, REVISION
from studio.storage import read, write


class Sandbox(unittest.TestCase):
    """Every test runs in its own temporary data directory."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in storage.FOLDERS:
            (self.root / name).mkdir()
        self.patches = [patch('studio.storage.DATA', self.root), patch('studio.management.DATA', self.root)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.temp.cleanup()

    def recording(self, key, clips=3, seconds=4.0):
        write(self.root / 'samples' / f'{key}.json', {'id': key, 'name': key + '.wav', 'seconds': seconds * clips})
        for i in range(clips):
            clip = f'{key}_{i:05d}'
            sf.write(self.root / 'clips' / f'{clip}.wav', np.sin(np.linspace(0, 400, int(16000 * seconds))) * .3, 16000)
            write(self.root / 'clips' / f'{clip}.json', {'id': clip, 'text': 'Hello there, this is a test sentence.', 'seconds': seconds, 'enabled': True})


class SettingsTests(unittest.TestCase):
    def test_validation_and_resume_rules(self):
        cfg = settings.validate({'lr_preset': 'standard', 'targets': ['v_proj', 'q_proj']})
        self.assertEqual(cfg['lr'], 1e-4)
        self.assertEqual(cfg['targets'], ['q_proj', 'v_proj'])  # Canonical order.
        with self.assertRaises(ValueError):
            settings.validate({'lr': 1})
        with self.assertRaises(ValueError):
            settings.validate({'targets': ['embed_tokens']})
        with self.assertRaises(ValueError):
            settings.validate({'gpus': [0, 1]}, gpu_count=2, distributed_ok=False)
        with self.assertRaises(ValueError):
            settings.validate({'gpus': [2]}, gpu_count=2)
        with self.assertRaises(ValueError):
            settings.validate({'unknown': 1})
        changed = settings.validate({**cfg, 'lr': 2e-4, 'batch_size': 4})
        settings.check_resume(cfg, changed)  # Resumable settings may change.
        with self.assertRaises(ValueError):
            settings.check_resume(cfg, settings.validate({**cfg, 'rank': 16}))

    def test_schedule(self):
        self.assertAlmostEqual(settings.learning_rate(1e-4, 0, 10, 2, 'constant'), 5e-5)
        self.assertAlmostEqual(settings.learning_rate(1e-4, 5, 10, 2, 'constant'), 1e-4)
        self.assertAlmostEqual(settings.learning_rate(1e-4, 10, 10, 0, 'linear'), 0)
        self.assertAlmostEqual(settings.learning_rate(1e-4, 10, 10, 0, 'cosine'), 1e-5)

    def test_lora_shapes_dropout_and_learning(self):
        torch.manual_seed(0)
        model = nn.ModuleDict({'self_attn': nn.ModuleDict({'q_proj': nn.Linear(6, 4), 'k_proj': nn.Linear(6, 2)})})
        params = adapters(model, rank=3, alpha=6, dropout=.1, targets=('q_proj', 'k_proj'))
        self.assertEqual(tuple(params['self_attn.k_proj.a'].shape), (3, 6))
        self.assertEqual(tuple(params['self_attn.k_proj.b'].shape), (2, 3))
        self.assertEqual(model['self_attn']['q_proj'].scale, 2.0)
        with self.assertRaises(ValueError):
            adapters(nn.ModuleDict({'x': nn.Linear(2, 2)}), targets=('q_proj',))


class SplitTests(unittest.TestCase):
    def rows(self, counts):
        return [{'id': f'{key}_{i:05d}', 'seconds': 5} for key, n in counts.items() for i in range(n)]

    def test_recording_level_split_is_persistent(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'split.json'
            cfg = settings.validate({'seed': 1})
            rows = self.rows({'aaaa': 10, 'bbbb': 3, 'cccc': 12})
            split, parts = kitten.build_split(path, rows, cfg)
            by_recording = {}
            for row in rows:
                by_recording.setdefault(row['id'].split('_')[0], set()).add(parts[row['id']])
            self.assertTrue(all(len(v) == 1 for v in by_recording.values()), 'a recording was split across partitions')
            self.assertEqual(split['recordings']['bbbb'], 'validation')  # Smallest recording held out first.
            write(path, split)
            split2, parts2 = kitten.build_split(path, rows + self.rows({'dddd': 4}), cfg)
            self.assertEqual({k: parts2[k] for k in parts}, parts)  # Existing members never move.

    def test_single_recording_has_no_validation_and_warns(self):
        with tempfile.TemporaryDirectory() as folder:
            split, parts = kitten.build_split(Path(folder) / 's.json', self.rows({'aaaa': 8}), settings.validate({}))
            self.assertEqual(set(parts.values()), {'train'})
            self.assertTrue(kitten.split_warning([], 'recording'))
            self.assertIn('Tiny', kitten.split_warning([{'seconds': 3}], 'recording')[0])

    def test_legacy_clip_split_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'split.json'
            write(path, {'kind': 'clip-legacy', 'clips': {'aaaa_00000': 'validation', 'aaaa_00001': 'train'}})
            _, parts = kitten.build_split(path, self.rows({'aaaa': 2}), settings.validate({}))
            self.assertEqual(parts, {'aaaa_00000': 'validation', 'aaaa_00001': 'train'})


class RegistryTests(Sandbox):
    def fake_checkpoint(self, path, arch, step=3):
        expected = {k: v for k, v in self.expected(arch).items()}
        adapter = {k: torch.randn(*shape) * .01 for k, shape in expected.items()}
        torch.save({'backend': registry.KITTEN, 'base': BASE, 'revision': REVISION, 'weights': 'emb4', 'architecture': arch,
                    'adapter': adapter, 'optimizer': {'state': {}, 'param_groups': []}, 'step': step, 'epochs': 1,
                    'embedding': torch.randn(512), 'reference_text': 'Hello there.', 'validation_loss': 1.5}, path)
        return adapter

    def expected(self, arch):
        # Two-layer stand-in for the 28-layer base so tests stay fast and offline.
        shapes = {'q_proj': (8, 8), 'v_proj': (8, 4)}
        return {f'model.layers.{i}.self_attn.{t}.{p}': ((arch['rank'], shapes[t][0]) if p == 'a' else (shapes[t][1], arch['rank']))
                for i in range(2) for t in arch['targets'] for p in 'ab'}

    def test_create_select_isolate_rename_archive_delete_restore(self):
        self.recording('aaaa')
        a = registry.create('Voice A', 'default', 'Speaker 1', {'rank': 4})
        b = registry.create('Voice B', 'default', 'Speaker 1', {'rank': 8})
        self.assertNotEqual(a['id'], b['id'])
        with self.assertRaises(ValueError):
            registry.checkpoint_path(a, 'latest')
        folder = registry.root() / a['id'] / 'checkpoints'
        for name, step in (('latest.pt', 9), ('best.pt', 6), ('update-000003.pt', 3)):
            self.fake_checkpoint(folder / name, a['architecture'], step)
            registry.record_checkpoint(a, name, {'update': step, 'epochs': 1})
        a.update(latest={'name': 'latest.pt', 'update': 9}, best={'name': 'best.pt', 'update': 6})
        registry.save(a)
        self.assertEqual(registry.checkpoint_path(a, 'best').name, 'best.pt')
        self.assertEqual(registry.checkpoint_path(a, 'latest').name, 'latest.pt')
        self.assertEqual(registry.checkpoint_path(a, 'update-000003.pt').name, 'update-000003.pt')
        for bad in ('../best.pt', 'update-1.pt', 'other.pt'):
            with self.assertRaises(ValueError):
                registry.checkpoint_path(a, bad)
        with self.assertRaises(ValueError):  # B has no checkpoints of its own: no cross-experiment access.
            registry.checkpoint_path(registry.load(b['id']), 'latest')
        with self.assertRaises(ValueError):
            registry.load('../../etc')
        dup = registry.duplicate(a['id'])
        self.assertEqual(dup['architecture'], a['architecture'])
        self.assertEqual(dup['checkpoints'], {})
        self.assertEqual(registry.rename(a['id'], 'Renamed')['name'], 'Renamed')
        self.assertTrue(registry.set_archived(a['id'], True)['archived'])
        with self.assertRaises(ValueError):
            registry.delete(a['id'], 'wrong name')
        manifest = registry.delete(a['id'], 'Renamed')
        self.assertFalse((registry.root() / a['id']).exists())
        management.restore(manifest['id'])
        self.assertEqual(registry.checkpoint_path(registry.load(a['id']), 'best').name, 'best.pt')

    def test_prune_keeps_best_and_latest(self):
        e = registry.create('Prune', 'default', 'Speaker 1', {})
        folder = registry.root() / e['id'] / 'checkpoints'
        for step in (1, 2, 3):
            (folder / f'update-{step:06d}.pt').write_bytes(b'x')
            registry.record_checkpoint(e, f'update-{step:06d}.pt', {'update': step})
        for name in ('latest.pt', 'best.pt'):
            (folder / name).write_bytes(b'x')
            registry.record_checkpoint(e, name, {'update': 3})
        registry.prune(e, 1)
        self.assertEqual(sorted(e['checkpoints']), ['best.pt', 'latest.pt', 'update-000003.pt'])
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ['best.pt', 'latest.pt', 'update-000003.pt'])

    def test_legacy_migration_copies_once(self):
        self.recording('aaaa')
        legacy = self.root / 'models'
        self.fake_checkpoint(legacy / 'latest.pt', settings.LEGACY_ARCHITECTURE | {'targets': ['q_proj', 'v_proj']}, step=9)
        sf.write(legacy / 'reference.wav', np.zeros(16000), 16000)
        write(legacy / 'split.json', {'aaaa_00000': 'validation'})
        original = hashlib.sha256((legacy / 'latest.pt').read_bytes()).hexdigest()
        datasets.migrate()
        self.assertEqual(read(self.root / 'samples' / 'aaaa.json')['dataset_id'], 'default')
        e = registry.migrate_legacy()
        self.assertTrue(e['legacy'])
        self.assertEqual(e['updates'], 9)
        self.assertEqual(e['architecture'], settings.LEGACY_ARCHITECTURE)
        self.assertEqual(read(registry.root() / e['id'] / 'split.json')['kind'], 'clip-legacy')
        self.assertEqual(hashlib.sha256((legacy / 'latest.pt').read_bytes()).hexdigest(), original)  # Untouched.
        registry.delete(e['id'], e['name'])
        self.assertIsNone(registry.migrate_legacy())  # Deleted experiments are not re-imported.
        self.assertEqual(len(registry.listing()), 0)

    def test_checkpoint_validation_refuses_mismatches(self):
        e = registry.create('V', 'default', 'Speaker 1', {'rank': 4})
        path = self.root / 'c.pt'
        self.fake_checkpoint(path, e['architecture'])
        saved = torch.load(path, weights_only=True)
        with patch('studio.backends.kitten.expected_adapter', side_effect=self.expected):
            kitten.validate_checkpoint(saved, e)
            with self.assertRaises(ValueError):
                kitten.validate_checkpoint(saved, dict(e, architecture=dict(e['architecture'], rank=8)))
            with self.assertRaises(ValueError):
                kitten.validate_checkpoint(dict(saved, revision='0' * 40), e)
            bad = dict(saved, adapter=dict(saved['adapter']))
            bad['adapter'].popitem()
            with self.assertRaises(ValueError):
                kitten.validate_checkpoint(bad, e)

    def test_export_and_import_round_trip_and_rejections(self):
        self.recording('aaaa')
        e = registry.create('Export me', 'default', 'Speaker 1', {'rank': 2})
        folder = registry.root() / e['id']
        adapter = self.fake_checkpoint(folder / 'checkpoints' / 'best.pt', e['architecture'])
        registry.record_checkpoint(e, 'best.pt', {'update': 3, 'epochs': 1})
        e['best'] = {'name': 'best.pt', 'update': 3}
        registry.save(e)
        sf.write(folder / 'reference.wav', np.zeros(24000 * 3), 24000)
        (self.root / 'samples' / 'secret.wav').write_bytes(b'unrelated recording')
        with patch('studio.backends.kitten.expected_adapter', side_effect=self.expected):
            archive = export.export(e['id'], 'best')
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(set(z.namelist()), {'model.json', 'adapter.safetensors', 'reference.wav', 'README.txt', 'SHA256SUMS.txt', 'infer.py'})
                self.assertNotIn(b'optimizer', z.read('adapter.safetensors')[:4096])
            without = export.export(e['id'], 'best', include_reference=False)
            with zipfile.ZipFile(without) as z:
                self.assertNotIn('reference.wav', z.namelist())
            imported = export.import_archive(archive)
            self.assertFalse(imported['resumable'])
            loaded = torch.load(registry.checkpoint_path(imported, 'latest'), weights_only=True)
            self.assertTrue(all(torch.equal(loaded['adapter'][k], v) for k, v in adapter.items()))

            def rewrite(source, target, change):
                with zipfile.ZipFile(source) as src, zipfile.ZipFile(target, 'w') as dst:
                    for info in src.infolist():
                        data = src.read(info.filename)
                        name, data = change(info.filename, data)
                        if name:
                            dst.writestr(name, data)
                return target
            tampered = rewrite(archive, self.root / 't.zip', lambda n, d: (n, d + b'x' if n == 'adapter.safetensors' else d))
            with self.assertRaisesRegex(ValueError, 'Checksum'):
                export.import_archive(tampered)
            for evil in ('../evil.py', '/abs/model.json', 'sub/model.json', 'C:model.json', 'payload.pt'):
                bad = self.root / 'evil.zip'
                with zipfile.ZipFile(archive) as src, zipfile.ZipFile(bad, 'w') as dst:
                    for info in src.infolist():
                        dst.writestr(info.filename, src.read(info.filename))
                    dst.writestr(evil, b'malicious')
                with self.assertRaises(ValueError, msg=evil):
                    export.import_archive(bad)
        self.assertFalse((self.root.parent / 'evil.py').exists())


class ApiTests(Sandbox):
    def test_resume_refuses_fixed_and_unknown_settings(self):
        from app import app
        e = registry.create('API', 'default', 'Speaker 1', {'rank': 4})
        client, headers = app.test_client(), {'X-Studio': 'local'}
        with patch('app.DATA', self.root), patch('app.busy', return_value=False), patch('app.gpu_cards', return_value=[{}]):
            for run in ({'rank': 64}, {'targets': ['k_proj']}, {'nonsense': 1}, {'lr': 5}):
                response = client.post('/api/jobs', json={'action': 'train', 'experiment_id': e['id'], 'run': run}, headers=headers)
                self.assertEqual(response.status_code, 400, run)
            self.assertEqual(list((self.root / 'jobs').glob('*.json')), [])  # Nothing was started.
            response = client.get(f"/api/experiments/{e['id']}/export?checkpoint=latest")
            self.assertEqual(response.status_code, 400)  # No checkpoint yet: clear error, no archive.
            self.assertEqual(client.post(f"/api/experiments/{e['id']}", json={'action': 'delete', 'confirm': 'nope'},
                                         headers=headers).status_code, 400)


if __name__ == '__main__':
    unittest.main()
