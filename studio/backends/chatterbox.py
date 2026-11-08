"""Chatterbox backend: generation in a separate environment (.venv-chatterbox) via a worker process.

Chosen variant: ResembleAI/chatterbox, the original English ChatterboxTTS (0.5B Llama backbone),
pinned to revision 5bb1f6ee… with chatterbox-tts==0.1.7 and torch 2.6.0. It exposes CFG and
exaggeration controls and fits an 8 GB card (measured 3.4 GB peak on an RTX 2070).
"""
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .. import storage

WORKER = Path(__file__).with_name('chatterbox_worker.py')
MODEL = 'ResembleAI/chatterbox'
REVISION = '5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18'
SAMPLE_RATE = 24000

DESCRIPTION = ('Resemble AI Chatterbox (original English model, 0.5B, MIT license), pinned revision. '
               'Base-model inference only: it imitates a reference recording zero-shot. No weights change, '
               'so this is NOT training. Every output carries Resemble AI\'s Perth audio watermark. '
               'Upstream was archived as retired research in August 2026 and receives no updates.')

TRAINING_UNAVAILABLE = (
    'Fine-tuning is unavailable. The official repository (chatterbox-tts 0.1.7, archived August 2026) ships no '
    'training code, and no verified training path is implemented here. Finishing it requires: a T3 training loop '
    'with cross-entropy on S3Tokenizer speech tokens conditioned on voice-encoder embeddings and text; '
    'LoRA or 8-bit optimizer states so the 0.5B model fits in a single card\'s memory; checkpoint save/resume; '
    'and proof that reloaded T3 weights generate correctly. Reference cloning is available instead.')

CONTROLS = {
    'exaggeration': {'type': 'float', 'min': .25, 'max': 2.0, 'default': .5, 'help': 'Emotional intensity. Values above ~0.7 are more dramatic and can be unstable.'},
    'cfg_weight': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': .5, 'help': 'Classifier-free guidance. Lower (~0.3) slows pacing; 0 reduces accent transfer from the reference.'},
    'temperature': {'type': 'float', 'min': .05, 'max': 2.0, 'default': .8, 'help': 'Sampling temperature.'},
    'repetition_penalty': {'type': 'float', 'min': 1.0, 'max': 2.0, 'default': 1.2, 'help': 'Discourages repeated speech tokens.'},
    'min_p': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': .05, 'help': 'Discard tokens below this fraction of the top probability.'},
    'top_p': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 1.0, 'help': 'Nucleus sampling cut-off.'},
    'seed': {'type': 'int', 'min': 0, 'max': 2 ** 31 - 1, 'default': None, 'help': 'Seeds torch before generation for reproducibility.'},
}

_cache = {'at': 0, 'value': None}


def python():
    override = os.environ.get('STUDIO_CHATTERBOX_PYTHON')
    if override:
        return Path(override)
    venv = storage.ROOT / '.venv-chatterbox'
    return venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def call(request, timeout):
    folder = storage.DATA / 'jobs'
    key = uuid.uuid4().hex
    request = dict(request, result=str(folder / f'chatterbox-{key}.result.json'))
    path = folder / f'chatterbox-{key}.request.json'
    path.write_text(json.dumps(request), encoding='utf-8')
    try:
        # -I: isolated mode, so nothing from this checkout or the main .venv leaks into the worker.
        completed = subprocess.run([str(python()), '-I', str(WORKER), str(path)], capture_output=True, text=True,
                                   timeout=timeout, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        result_path = Path(request['result'])
        if not result_path.exists():
            raise RuntimeError('Chatterbox worker failed: ' + (completed.stderr or completed.stdout)[-800:])
        return json.loads(result_path.read_text(encoding='utf-8'))
    finally:
        path.unlink(missing_ok=True)
        Path(request['result']).unlink(missing_ok=True)


def availability():
    if time.time() - _cache['at'] < 300 and _cache['value']:
        return _cache['value']
    if not python().exists():
        value = {'available': False, 'reason': 'Chatterbox environment not installed. Run scripts/setup-chatterbox.bat '
                                                '(Windows) or bash scripts/setup-chatterbox.sh (Linux).'}
    else:
        try:
            result = call({'mode': 'check'}, timeout=120)
            if not result.get('ok'):
                value = {'available': False, 'reason': result.get('error', 'Import failed.')}
            elif not result.get('cuda'):
                value = {'available': False, 'reason': 'CUDA is unavailable in .venv-chatterbox.'}
            else:
                value = {'available': True, 'reason': '', 'versions': result.get('versions')}
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
            value = {'available': False, 'reason': str(exc)[:300]}
    _cache.update(at=time.time(), value=value)
    return value


def generate(items, timeout=3600):
    """items: [{'text', 'out', 'reference', **controls}] -> worker result."""
    result = call({'mode': 'generate', 'device': 'cuda', 'items': items}, timeout=timeout)
    if not result.get('ok'):
        raise RuntimeError('Chatterbox generation failed: ' + result.get('error', 'unknown error'))
    return result


if __name__ == '__main__':
    print(json.dumps(availability(), indent=2))
    sys.exit(0)
