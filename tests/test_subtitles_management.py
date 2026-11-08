import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
from studio import management
from studio.storage import read, write
from studio.subtitles import parse, grouped, word_error
from app import app

SRT = '1\n00:00:00,000 --> 00:00:02,000\nHello world.\n\n2\n00:00:02,100 --> 00:00:04,100\nThis is a test.\n'


class SubtitleTests(unittest.TestCase):
    def test_parse_and_group(self):
        rows = parse('\ufeff' + SRT, 5)
        result = grouped(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['text'], 'Hello world. This is a test.')
        self.assertEqual(result[0]['end'], 4.1)
        self.assertEqual(word_error('Hello, WORLD!', 'hello world'), 0)
        self.assertEqual(word_error('one two three', 'one three'), 1/3)
        with self.assertRaises(ValueError):
            parse(SRT, 3)
        with self.assertRaises(ValueError):
            parse(SRT.replace('00:00:02,100', '00:00:01,100'), 5)

    def test_import_delete_restore_and_purge(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for kind in ('samples', 'clips', 'features', 'models', 'outputs', 'subtitles'):
                (root / kind).mkdir()
            key = 'a' * 24
            sf.write(root / 'samples' / (key + '.wav'), np.zeros(80000), 16000)
            write(root / 'samples' / (key + '.json'), {'id': key, 'seconds': 5, 'name': 'test.wav'})
            old = root / 'clips' / (key + '_00000.json')
            write(old, {'text': 'old'})
            headers = {'X-Studio': 'local'}
            with patch('app.DATA', root), patch('studio.management.DATA', root), patch('studio.locking.DATA', root), patch('app.busy', return_value=False):
                client = app.test_client()
                # Reject malformed SRT before moving any existing clips.
                response = client.post('/api/subtitles/' + key, data={'file': (io.BytesIO(b'invalid'), 'test.srt')}, headers=headers)
                self.assertEqual(response.status_code, 400)
                self.assertTrue(old.exists())
                response = client.post('/api/subtitles/' + key, data={'file': (io.BytesIO(SRT.encode()), 'test.srt')}, headers=headers)
                self.assertEqual(response.status_code, 200, response.get_json())
                self.assertFalse(old.exists())
                clip = list((root / 'clips').glob('*.json'))[0]
                self.assertFalse(read(clip)['enabled'])
                self.assertEqual(read(clip)['source'], 'srt')
                self.assertAlmostEqual(sf.info(clip.with_suffix('.wav')).duration, 4.1, places=3)
                from studio.engine import check_subtitles
                expected_text = read(clip)['text']
                def fake_pipeline(*args, **kwargs):
                    return lambda *a, **kw: {'text': 'Hello world. This is a test.'}
                with patch('studio.engine.DATA', root), patch('studio.storage.DATA', root), patch('studio.engine.gpu'), patch('studio.engine.load_transcriber', side_effect=fake_pipeline):
                    check_subtitles(lambda *a, **kw: None, lambda: False, sample_id=key)
                self.assertTrue(read(clip)['enabled'])
                self.assertEqual(read(clip)['text'], expected_text)
                with patch('studio.engine.DATA', root), patch('studio.storage.DATA', root), patch('studio.engine.gpu'), patch('studio.engine.load_transcriber', return_value=lambda *a, **kw: {'text': 'Completely different words'}):
                    check_subtitles(lambda *a, **kw: None, lambda: False, sample_id=key)
                self.assertFalse(read(clip)['enabled'])
                self.assertEqual(read(clip)['review'], 'needs_review')
                self.assertEqual(read(clip)['text'], expected_text)
                deleted = client.post('/api/manage', json={'kind': 'samples', 'key': key}, headers=headers)
                self.assertEqual(deleted.status_code, 200)
                self.assertFalse(clip.exists())
                trash_id = deleted.get_json()['id']
                restored = client.post('/api/manage', json={'action': 'restore', 'key': trash_id}, headers=headers)
                self.assertEqual(restored.status_code, 200)
                self.assertTrue(clip.exists())
                # Refuse to overwrite newer files during restore.
                second = management.trash(management.collect('clips', clip.stem), 'test')
                write(clip, {'text': 'new'})
                with self.assertRaises(ValueError):
                    management.restore(second['id'])
                blocked = client.post('/api/manage', json={'action': 'purge', 'key': second['id']}, headers=headers)
                self.assertEqual(blocked.status_code, 400)
                purged = client.post('/api/manage', json={'action': 'purge', 'key': second['id'], 'confirm': 'DELETE'}, headers=headers)
                self.assertEqual(purged.status_code, 200)
                self.assertTrue(clip.exists())
                with self.assertRaises(ValueError):
                    management.collect('samples', '../outside')


if __name__ == '__main__':
    unittest.main()
