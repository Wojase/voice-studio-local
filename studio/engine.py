"""Local KittenTTS2 adaptation. Frozen base + trainable attention LoRA weights."""
import gc
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from .storage import DATA, read, write, clips
from . import distributed as parallel

BASE = 'KittenML/kitten-tts-2'
REVISION = 'f2eee74f4467855ad58fe16f1fb7b7df473263fd'


def gpu():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. NVIDIA GPU and CUDA-enabled PyTorch are required.')
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    torch.backends.cudnn.benchmark = True


def repository():
    from huggingface_hub import snapshot_download, hf_hub_download
    from kittenml.get_model import _kitten2_files
    config = read(Path(hf_hub_download(BASE, 'config.json', revision=REVISION)))
    path = snapshot_download(BASE, revision=REVISION, allow_patterns=_kitten2_files(config, weights='emb4'))
    return path, config


def load_model(training=False):
    from kittenml.kittentts2.model import KittenTTS2
    path, config = repository()
    # Preserve checkpoint range: FP16 overflows in attention layer 23 on real audio.
    model = KittenTTS2.__new__(KittenTTS2)
    if training:
        model.repo_dir = Path(path)
        model.config = config
        model.device = 'cpu'
        model.weights = 'emb4'
        model._load_language_model()
    else:
        model = KittenTTS2(path, config, device='cpu', weights='emb4')
    model.model.bfloat16().cuda()
    model.device, model.dtype = 'cuda', torch.bfloat16
    if not training:
        model.codec.s3gen.cuda()
        model.codec.device = 'cuda'
        model.speaker_encoder.device = 'cuda'
    return model


class LoRA(nn.Module):
    """Frozen base linear + trainable low-rank residual. Parameter names `a`/`b` match legacy checkpoints."""
    def __init__(self, base, rank=8, alpha=16.0, dropout=0.0):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x):
        residual = nn.functional.linear(nn.functional.linear(self.dropout(x.float()), self.a), self.b)
        return self.base(x) + (residual * self.scale).to(x.dtype)


def adapters(model, rank=8, alpha=16.0, dropout=0.0, targets=('q_proj', 'v_proj')):
    model.requires_grad_(False)
    found = 0
    for name, layer in list(model.named_modules()):
        if isinstance(layer, nn.Linear) and name.split('.')[-1] in targets:
            parent, key = name.rsplit('.', 1)
            setattr(model.get_submodule(parent), key, LoRA(layer, rank, alpha, dropout))
            found += 1
    if not found:
        raise ValueError('No layers matched the LoRA targets.')
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def load_transcriber():
    from transformers import pipeline
    return pipeline('automatic-speech-recognition',
                    model=os.environ.get('STUDIO_STT_MODEL', 'openai/whisper-small.en'),
                    device=0, dtype=torch.float16)


def recognize(transcriber, audio):
    model_name = os.environ.get('STUDIO_STT_MODEL', 'openai/whisper-small.en')
    # English-only checkpoints reject multilingual language/task arguments.
    kwargs = {} if model_name.endswith('.en') else {'language': 'en', 'task': 'transcribe'}
    return transcriber(audio, generate_kwargs=kwargs)['text'].strip()


def prepare(report, stopped, **_):
    gpu()
    import librosa
    sources = sorted((DATA / 'samples').glob('*.wav'))
    if not sources:
        raise ValueError('Upload recordings first.')
    pending = []
    for source in sources:
        if (DATA / 'subtitles' / (source.stem + '.json')).exists():
            continue  # User-supplied timed text is authoritative; never overwrite it with ASR.
        # Stream long recordings in bounded chunks rather than loading hours into RAM.
        with sf.SoundFile(source) as audio:
            segment = 12 * audio.samplerate
            for index, start in enumerate(range(0, len(audio), segment)):
                key = f'{source.stem}_{index:05d}'
                meta = DATA / 'clips' / f'{key}.json'
                if meta.exists():
                    continue
                audio.seek(start)
                wav = audio.read(segment, dtype='float32', always_2d=True).mean(axis=1)
                if len(wav) < audio.samplerate * 2 or np.sqrt(np.mean(wav ** 2)) < .002:
                    continue
                dest = DATA / 'clips' / f'{key}.wav'
                sf.write(dest, wav, audio.samplerate, subtype='PCM_16')
                pending.append((dest, meta, len(wav) / audio.samplerate))
    if pending:
        report('Loading English speech recognition on GPU...')
        transcriber = load_transcriber()
        for i, (wav, meta, seconds) in enumerate(pending):
            if stopped():
                return
            audio, _ = librosa.load(wav, sr=16000)
            text = recognize(transcriber, audio)
            if text:
                write(meta, {'id': wav.stem, 'text': text, 'seconds': seconds, 'enabled': True})
            report(f'Transcribing {i + 1}/{len(pending)}', progress=(i + 1) / len(pending))
        del transcriber
        gc.collect()
        torch.cuda.empty_cache()
    report('Clips prepared. Review transcripts before training.', progress=1)


def check_subtitles(report, stopped, sample_id='', sample_ids=None, **_):
    """Compare supplied timed text against ASR; never replace the supplied transcript."""
    gpu()
    import librosa
    from .subtitles import word_error
    from .management import safe_key
    ids = set(safe_key(key) for key in (sample_ids or [sample_id]))
    selected = [p for p in clips() if read(p).get('sample_id') in ids and read(p).get('source') == 'srt']
    if not selected:
        raise ValueError('No subtitle clips found for this recording.')
    report('Loading English speech recognition for subtitle checking…')
    transcriber = load_transcriber()
    for i, path in enumerate(selected):
        if stopped():
            return
        row = read(path)
        audio, _ = librosa.load(path.with_suffix('.wav'), sr=16000)
        hypothesis = recognize(transcriber, audio)
        error = word_error(row['text'], hypothesis)
        row['check'] = {'recognized': hypothesis, 'word_error': error,
                        'model': os.environ.get('STUDIO_STT_MODEL', 'openai/whisper-small.en')}
        row['review'] = 'match' if error <= .25 else 'needs_review'
        # A mismatch is excluded until manually reviewed; ASR agreement isn't proof of accuracy.
        row['enabled'] = error <= .25
        write(path, row)
        report(f'Subtitle check {i + 1}/{len(selected)} · word error {error:.0%}', progress=(i + 1) / len(selected))
    report('Subtitle check complete. Review flagged clips before training.', progress=1)


def features(report, stopped, records=None):
    """Cache base-model codec tokens and speaker embeddings per clip (shared by experiments on this base)."""
    from kittenml.kittentts2.vocoder import S3Codec
    from kittenml.kittentts2.speaker import SpeakerEncoder
    if records is None:
        records = [read(p) for p in clips() if read(p).get('enabled', True)]
    if len(records) < 2:
        raise ValueError('At least two enabled clips are required. Aim for 20-60 minutes of clean speech.')
    missing = [r for r in records if not (DATA / 'features' / (r['id'] + '.pt')).exists()]
    if missing:
        report('Encoding new clips on GPU...')
        repo, config = repository()
        codec = S3Codec(device='cuda')
        speaker = SpeakerEncoder(Path(repo) / config['speaker_embedding'], device='cuda')
        for i, row in enumerate(missing):
            if stopped():
                return []
            wav = DATA / 'clips' / (row['id'] + '.wav')
            values = {'tokens': codec.encode_reference(wav), 'embedding': torch.tensor(speaker(wav))}
            torch.save(values, DATA / 'features' / (row['id'] + '.pt'))
            report(f'Encoding {i + 1}/{len(missing)}')
        del codec, speaker
        gc.collect()
        torch.cuda.empty_cache()
    return records


def train(report, stopped, **args):
    from .backends import kitten
    return kitten.train(report, stopped, **args)


def generate(report, stopped, **args):
    from . import generation
    return generation.run(report, stopped, **args)
