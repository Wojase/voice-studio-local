import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import soundfile as sf
from studio.distributed import groups
from studio.media import convert


class ScalingMediaTests(unittest.TestCase):
    def test_shards_and_equal_update_counts(self):
        first = list(groups(list(range(19)), 2, 3, 0, 2))
        second = list(groups(list(range(19)), 2, 3, 1, 2))
        self.assertEqual(len(first), len(second))
        self.assertEqual([len(b) for _, b in first], [2, 2])
        self.assertFalse(set(sum(first[0][1], [])) & set(sum(second[0][1], [])))
        self.assertEqual(set(sum([sum(b, []) for _, b in first + second], [])), set(range(19)))

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg required')
    def test_real_mp4_audio_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'test with spaces.mp4'
            subprocess.run(['ffmpeg', '-loglevel', 'error', '-f', 'lavfi', '-i',
                            'sine=frequency=440:duration=2.5', '-c:a', 'aac', str(source)], check=True)
            mp3 = convert(source, root / 'out.mp3', mp3=True)
            wav = convert(source, root / 'out.wav')
            self.assertGreater(mp3.stat().st_size, 1000)
            self.assertEqual(sf.info(wav).samplerate, 24000)
            self.assertGreater(sf.info(wav).duration, 2)
            before = mp3.read_bytes()
            with self.assertRaises(FileExistsError):
                convert(source, mp3, mp3=True)
            self.assertEqual(mp3.read_bytes(), before)
            self.assertTrue(source.exists())
            from app import app
            (root / 'samples').mkdir()
            with patch('app.DATA', root), patch('app.busy', return_value=False):
                with source.open('rb') as stream:
                    response = app.test_client().post('/api/upload', data={'file': (stream, 'movie.MP4')},
                                                      headers={'X-Studio': 'local'})
                self.assertEqual(response.status_code, 200, response.get_json())
                self.assertEqual(len(list((root / 'converted').glob('*.mp3'))), 1)
                self.assertEqual(len(list((root / 'samples').glob('*.wav'))), 1)
