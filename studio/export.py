"""Portable adapter archives and a validated, pickle-free import.

An export contains ONLY: the selected checkpoint's adapter weights and speaker embedding
(safetensors), model.json, an optional disclosed reference clip, an inference script, a README
and checksums. Never optimizer state, base weights, other recordings or other experiments.
"""
import hashlib
import io
import json
import re
import time
import uuid
import zipfile
from pathlib import Path

import soundfile as sf

from . import storage, registry, config as settings
from .storage import read, write, now

FORMAT = 'voice-studio-export/1'
ALLOWED = {'model.json', 'adapter.safetensors', 'reference.wav', 'README.txt', 'SHA256SUMS.txt', 'infer.py'}
MAX_MEMBER = 2 * 1024 ** 3
MAX_ARCHIVE_MEMBERS = 16

INFER = r'''"""Standalone inference for a Voice Studio KittenTTS2 LoRA adapter.

pip install kittenml==0.9.4 safetensors soundfile   (plus CUDA-enabled PyTorch)
python infer.py "Hello, this is my voice." out.wav
The base model KittenML/kitten-tts-2 is downloaded from Hugging Face at the pinned revision; it is not included.
"""
import json, math, sys
from pathlib import Path
import soundfile as sf, torch
from torch import nn
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download, snapshot_download
from kittenml.get_model import _kitten2_files
from kittenml.kittentts2.model import KittenTTS2

here = Path(__file__).parent
meta = json.loads((here / 'model.json').read_text(encoding='utf-8'))
arch, base = meta['architecture'], meta['base']


class LoRA(nn.Module):
    def __init__(self, layer, rank, alpha):
        super().__init__()
        self.base = layer
        self.a = nn.Parameter(torch.zeros(rank, layer.in_features, device=layer.weight.device))
        self.b = nn.Parameter(torch.zeros(layer.out_features, rank, device=layer.weight.device))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (nn.functional.linear(nn.functional.linear(x.float(), self.a), self.b) * self.scale).to(x.dtype)


config = json.loads(Path(hf_hub_download(base['id'], 'config.json', revision=base['revision'])).read_text())
path = snapshot_download(base['id'], revision=base['revision'], allow_patterns=_kitten2_files(config, weights=base['weights']))
tts = KittenTTS2(path, config, device='cpu', weights=base['weights'])
tts.model.bfloat16().cuda(); tts.device, tts.dtype = 'cuda', torch.bfloat16
tts.codec.s3gen.cuda(); tts.codec.device = 'cuda'; tts.speaker_encoder.device = 'cuda'
tts.model.requires_grad_(False)
for name, layer in list(tts.model.named_modules()):
    if isinstance(layer, nn.Linear) and name.split('.')[-1] in arch['targets']:
        parent, key = name.rsplit('.', 1)
        setattr(tts.model.get_submodule(parent), key, LoRA(layer, arch['rank'], arch['alpha']))
tensors = load_file(str(here / 'adapter.safetensors'))
params = dict(tts.model.named_parameters())
for name, value in tensors.items():
    if name != 'speaker_embedding':
        params[name].data.copy_(value)
reference = here / 'reference.wav'
if not reference.exists():
    sys.exit('reference.wav was not included in this export. Pass any clean clip of the same speaker as reference.wav.')
tts.model.eval()
tts._reference_cache[('voice', 'local')] = tts._build_reference(tensors['speaker_embedding'].numpy(), [], meta['reference_text'], str(reference))
tts.voices['local'] = {}
audio = tts.generate(sys.argv[1], voice='local', use_reference_prompt=False)
sf.write(sys.argv[2] if len(sys.argv) > 2 else 'out.wav', audio, tts.sample_rate)
'''


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def export(experiment_id, which='best', include_reference=True):
    """Write the archive to exports/ (file-backed) and return its path."""
    from .backends.kitten import load_checkpoint, validate_checkpoint
    from safetensors.torch import save as save_tensors
    experiment = registry.load(experiment_id)
    path = registry.checkpoint_path(experiment, which)
    saved = load_checkpoint(path)
    architecture = validate_checkpoint(saved, experiment)
    tensors = {k: v.contiguous() for k, v in saved['adapter'].items()}
    tensors['speaker_embedding'] = saved['embedding'].float().contiguous()
    adapter = save_tensors(tensors, metadata={'format': FORMAT})
    reference = registry.root() / experiment_id / 'reference.wav'
    include_reference = include_reference and reference.is_file()
    meta = {
        'format': FORMAT, 'type': 'adapter', 'standalone': False,
        'requires': 'Separately obtained base weights: KittenML/kitten-tts-2 at the revision below (downloaded from '
                    'Hugging Face by infer.py). Base weights are NOT included; they are subject to the KittenML model license.',
        'backend': registry.KITTEN, 'base': experiment['base'], 'architecture': architecture,
        'adapter_scale': architecture['alpha'] / architecture['rank'],
        'experiment': {'id': experiment['id'], 'name': experiment['name'], 'speaker': experiment['speaker']},
        'checkpoint': {'requested': which, 'file': path.name, 'update': saved['step'], 'epochs': saved['epochs'],
                       'validation_loss': saved.get('validation_loss'), 'best_validation_loss': saved.get('best_loss')},
        'sample_rate': 24000, 'language': 'en',
        'preprocessing': {'clips': 'mono PCM16 WAV, ~2-12 s, original sample rate', 'features': 'S3 codec audio tokens + 512-d speaker embedding',
                          'prompt': 'kittenml build_generation_prompt, max 1024 tokens', 'precision': 'bfloat16 base, float32 adapters'},
        'reference_text': saved['reference_text'], 'reference_included': include_reference,
        'training_config': saved.get('config') or experiment.get('config'),
        'software': experiment.get('software'), 'exported': now(),
        'contents_note': 'Contains only this checkpoint. No optimizer state, training recordings or other experiments.',
    }
    files = {'model.json': json.dumps(meta, indent=2, ensure_ascii=False).encode(), 'adapter.safetensors': adapter,
             'infer.py': INFER.encode()}
    readme = (f"Voice Studio adapter export: {experiment['name']} / {path.name} (update {saved['step']})\n\n"
              'TYPE: LoRA adapter. Not a standalone model; it requires the KittenML/kitten-tts-2 base weights at '
              f"revision {experiment['base']['revision']}, which are NOT included.\n"
              + ('INCLUDED VOICE DATA: reference.wav is a short clip of the trained speaker, used for decoder conditioning.\n'
                 if include_reference else 'reference.wav is NOT included. Provide a clean clip of the same speaker.\n') +
              '\nFresh machine:\n  python -m venv venv && venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128\n'
              '  venv/bin/pip install kittenml==0.9.4 safetensors soundfile\n  venv/bin/python infer.py "Hello there." out.wav\n'
              '\nOr import this archive into Voice Studio (Models > Import). Verify: sha256sum -c SHA256SUMS.txt\n')
    files['README.txt'] = readme.encode()
    sums = {name: sha256_bytes(data) for name, data in files.items()}
    if include_reference:
        sums['reference.wav'] = registry.sha256(reference)
    files['SHA256SUMS.txt'] = ''.join(f'{d}  {n}\n' for n, d in sorted(sums.items())).encode()
    safe = re.sub(r'[^A-Za-z0-9_-]+', '-', experiment['name']).strip('-')[:40] or 'experiment'
    target = storage.DATA / 'exports' / f"{safe}-{path.stem}-u{saved['step']}-{time.strftime('%Y%m%d-%H%M%S')}.zip"
    partial = target.with_suffix('.partial')
    with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        if include_reference:
            archive.write(reference, 'reference.wav', compress_type=zipfile.ZIP_STORED)
    partial.replace(target)
    return target


def safe_members(archive):
    infos = archive.infolist()
    if len(infos) > MAX_ARCHIVE_MEMBERS:
        raise ValueError('Archive has too many entries.')
    names = set()
    for info in infos:
        name = info.filename
        # Flat archive only: rejects absolute paths, drive letters, '..', backslashes and directories.
        if name not in ALLOWED or '/' in name or '\\' in name or name.startswith('.') or ':' in name:
            raise ValueError(f'Unexpected or unsafe archive entry: {name!r}')
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError('Symbolic links are not allowed in archives.')
        if info.file_size > MAX_MEMBER or info.compress_size and info.file_size / info.compress_size > 200:
            raise ValueError(f'Archive entry {name} is too large.')
        if name in names:
            raise ValueError('Duplicate archive entry.')
        names.add(name)
    for required in ('model.json', 'adapter.safetensors', 'SHA256SUMS.txt'):
        if required not in names:
            raise ValueError(f'Archive is missing {required}.')
    return names


def import_archive(stream_path, name=None, dataset_id=None, speaker=None):
    """Validate an export and register it as a new, inference-only experiment. Never unpickles anything."""
    from safetensors.torch import load as load_tensors
    from .backends.kitten import validate_checkpoint
    from . import datasets
    with zipfile.ZipFile(stream_path) as archive:
        names = safe_members(archive)
        data = {n: archive.read(n) for n in names}
    sums = {}
    for line in data['SHA256SUMS.txt'].decode().splitlines():
        if line.strip():
            digest, member = line.split(None, 1)
            sums[member.strip()] = digest
    for member in names - {'SHA256SUMS.txt'}:
        if sums.get(member) != sha256_bytes(data[member]):
            raise ValueError(f'Checksum mismatch for {member}.')
    meta = json.loads(data['model.json'])
    if meta.get('format') != FORMAT or meta.get('backend') != registry.KITTEN:
        raise ValueError('Unsupported archive format or backend.')
    if meta.get('base') != registry.base_identity():
        raise ValueError('Archive was trained on a different base model or revision.')
    architecture = settings.architecture(settings.validate(meta.get('architecture', {})))
    if architecture != meta['architecture']:
        raise ValueError('Archive adapter architecture is not valid.')
    tensors = load_tensors(data['adapter.safetensors'])
    embedding = tensors.pop('speaker_embedding', None)
    checkpoint = {'backend': registry.KITTEN, 'base': meta['base']['id'], 'revision': meta['base']['revision'],
                  'weights': meta['base']['weights'], 'architecture': architecture, 'adapter': tensors, 'embedding': embedding}
    validate_checkpoint(checkpoint, {'architecture': architecture})
    if 'reference.wav' in data:
        info = sf.info(io.BytesIO(data['reference.wav']))
        if info.duration < 1 or info.duration > 60:
            raise ValueError('Included reference clip has an implausible length.')
    dataset_id = dataset_id or datasets.DEFAULT_ID
    speaker = speaker or datasets.find(dataset_id)['speakers'][0]
    cfg = dict(meta.get('training_config') or {})
    cfg = settings.validate({**{k: v for k, v in cfg.items() if k in settings.FIXED or k in settings.RESUMABLE}, **architecture})
    experiment = registry.create(name or f"Imported: {meta['experiment']['name']}"[:80], dataset_id, speaker, cfg,
                                 imported=True, resumable=False, import_source={'file': name, 'meta': meta['checkpoint'], 'at': now()})
    folder = registry.root() / experiment['id']
    import torch
    torch.save({**checkpoint, 'step': meta['checkpoint']['update'], 'epochs': meta['checkpoint']['epochs'],
                'reference_text': meta['reference_text'], 'validation_loss': meta['checkpoint'].get('validation_loss'),
                'format': 'voice-studio/kittentts2-lora/2'}, folder / 'checkpoints' / 'latest.pt')
    if 'reference.wav' in data:
        (folder / 'reference.wav').write_bytes(data['reference.wav'])
    registry.record_checkpoint(experiment, 'latest.pt', {'update': meta['checkpoint']['update'], 'epochs': meta['checkpoint']['epochs'],
                                                         'validation_loss': meta['checkpoint'].get('validation_loss'), 'saved': now(), 'source': 'import'})
    experiment.update(latest={'name': 'latest.pt', 'update': meta['checkpoint']['update']}, status='ready',
                      updates=meta['checkpoint']['update'], epochs=meta['checkpoint']['epochs'],
                      reference={'clip_id': None, 'text': meta['reference_text']})
    registry.save(experiment)
    registry.log(experiment['id'], f'Imported adapter archive ({meta["experiment"]["name"]}, update {meta["checkpoint"]["update"]}). Inference only.')
    return experiment
