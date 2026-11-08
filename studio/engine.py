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
    def __init__(self, base, rank=8):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = 2.0

    def forward(self, x):
        residual = nn.functional.linear(nn.functional.linear(x.float(), self.a), self.b)
        return self.base(x) + (residual * self.scale).to(x.dtype)


def adapters(model):
    model.requires_grad_(False)
    for name, layer in list(model.named_modules()):
        if isinstance(layer, nn.Linear) and name.split('.')[-1] in ('q_proj', 'v_proj'):
            parent, key = name.rsplit('.', 1)
            setattr(model.get_submodule(parent), key, LoRA(layer))
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


def check_subtitles(report, stopped, sample_id='', **_):
    """Compare supplied timed text against ASR; never replace the supplied transcript."""
    gpu()
    import librosa
    from .subtitles import word_error
    from .management import safe_key
    safe_key(sample_id)
    selected = [p for p in clips() if read(p).get('sample_id') == sample_id and read(p).get('source') == 'srt']
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


def features(report, stopped):
    from kittenml.kittentts2.vocoder import S3Codec
    from kittenml.kittentts2.speaker import SpeakerEncoder
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


def save_checkpoint(path, value):
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def train(report, stopped, epochs=3, accumulation=4, lr=0.0001, batch_size=1, **_):
    gpu()
    torch.manual_seed(42)
    if parallel.rank() == 0:
        features(report, stopped)
    parallel.barrier()
    records = [read(p) for p in clips() if read(p).get('enabled', True)]
    if parallel.should_stop(stopped):
        return
    torch.manual_seed(42)  # Feature extraction on rank 0 must not change LoRA initialization.
    report('Loading training model...')
    tts = load_model(training=True)
    model = tts.model
    params = adapters(model)
    if parallel.world() > 1:
        for parameter in params.values():
            torch.distributed.broadcast(parameter.data, src=0)
    optimizer = torch.optim.AdamW(list(params.values()), lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    checkpoint = DATA / 'models' / 'latest.pt'
    step, completed_epochs, best_loss = 0, 0, float('inf')
    if checkpoint.exists():
        previous = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if previous['base'] != BASE or previous['weights'] != 'emb4' or previous['revision'] != REVISION:
            raise ValueError('Checkpoint base model does not match.')
        for name, value in previous['adapter'].items():
            params[name].data.copy_(value)
        optimizer.load_state_dict(previous['optimizer'])
        for group in optimizer.param_groups:
            group['lr'] = lr
        scaler.load_state_dict(previous['scaler'])
        step, completed_epochs = previous['step'], previous['epochs']
        best_loss = previous.get('best_loss', float('inf'))
        del previous
    from kittenml.kittentts2.prompt import build_generation_prompt
    from kittenml.kittentts2.tokens import TokenMap
    tm = TokenMap.from_config(tts.config)
    examples = []
    embeddings = []
    for row in records:
        feature = torch.load(DATA / 'features' / (row['id'] + '.pt'), weights_only=True)
        prompt = build_generation_prompt(tm, tts.tokenizer.encode(row['text'], add_special_tokens=False))
        target = [tm.audio_id_base + t for t in feature['tokens']] + [tm.speech_end_id]
        if len(prompt) + len(target) > 1024:
            raise ValueError(f"Clip is too long: {row['id']}. Shorten the transcript or disable this clip.")
        examples.append((prompt, target, feature['embedding']))
        embeddings.append(feature['embedding'])
    voice_embedding = torch.stack(embeddings).mean(0)
    voice_embedding = nn.functional.normalize(voice_embedding, dim=0)
    reference = DATA / 'models' / 'reference.wav'
    if not reference.exists():
        shutil.copyfile(DATA / 'clips' / (records[0]['id'] + '.wav'), reference)
    # Non-reentrant checkpointing supports frozen embedding inputs and trainable LoRA.
    if os.environ.get('STUDIO_CHECKPOINTING', '1') == '1':
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    model.train()
    randomizer = random.Random(42 + step)
    indices = list(range(len(examples)))
    random.Random(42).shuffle(indices)
    holdout = indices[:max(1, len(indices) // 10)] if len(indices) >= 10 else []
    training = indices[len(holdout):]
    # Persist memberships so adding recordings does not move old training clips
    # into validation and produce misleading validation scores.
    split_path = DATA / 'models' / 'split.json'
    split = read(split_path, {})
    if not split:
        split = {records[i]['id']: ('validation' if i in holdout else 'train') for i in indices}
    for row in records:
        if row['id'] not in split:
            split[row['id']] = 'validation' if int(row['id'][:8], 16) % 10 == 0 else 'train'
    training = [i for i in indices if split[records[i]['id']] == 'train']
    holdout = [i for i in indices if split[records[i]['id']] == 'validation']
    if not training:
        raise ValueError('Enable at least one training clip.')
    if parallel.rank() == 0:
        write(split_path, split)

    def loss_for(batch):
        selected = [examples[index] for index in batch]
        length = max(len(p) + len(t) - 1 for p, t, _ in selected)
        ids = torch.zeros((len(batch), length), dtype=torch.long, device='cuda')
        mask = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        for i, (prompt, target, _) in enumerate(selected):
            size = len(prompt) + len(target) - 1
            ids[i, :size] = torch.tensor(prompt + target[:-1], device='cuda')
            mask[i, :size] = 1
            labels[i, len(prompt) - 1:size] = torch.tensor(target, device='cuda')
        # Same speaker conditioning as inference; only predict target audio.
        with torch.autocast('cuda', dtype=torch.bfloat16):
            inputs = model.get_input_embeddings()(ids)
            embeddings_batch = torch.stack([e for _, _, e in selected]).to('cuda', dtype=torch.bfloat16)
            projected = model.spk_proj(embeddings_batch).unsqueeze(1)
            inputs = torch.cat([projected, inputs[:, 1:]], dim=1)
            hidden = model.model(inputs_embeds=inputs, attention_mask=mask, use_cache=False).last_hidden_state
            valid = labels != -100
            # Avoid allocating vocabulary logits for text and padding positions.
            logits = model.lm_head(hidden[valid])
            return nn.functional.cross_entropy(logits.float(), labels[valid])

    def save(epoch_count):
        if parallel.rank() != 0:
            return
        save_checkpoint(checkpoint, {'base': BASE, 'revision': REVISION, 'weights': 'emb4', 'adapter': {k: v.detach().cpu() for k, v in params.items()},
                                    'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                                    'step': step, 'epochs': epoch_count, 'best_loss': best_loss, 'embedding': voice_embedding,
                                    'reference_text': records[0]['text']})
        write(DATA / 'models' / 'info.json', {'steps': step, 'epochs': epoch_count, 'samples': len(records),
                                            'validation_samples': len(holdout), 'base': BASE})

    for epoch in range(epochs):
        randomizer.shuffle(training)
        for offset, batches in parallel.groups(training, accumulation, batch_size, parallel.rank(), parallel.world()):
            if parallel.should_stop(stopped):
                save(completed_epochs + epoch)
                return
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for micro, batch in enumerate(batches):
                report(f'Epoch {epoch + 1}/{epochs} · batch {offset + micro + 1}/{len(training)} · saved steps {step}',
                       progress=(epoch + (offset + micro) / len(training)) / epochs)
                loss = loss_for(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError('Training became unstable. The last completed checkpoint is preserved.')
                total += loss.item()
                scaler.scale(loss / len(batches)).backward()
            scaler.unscale_(optimizer)
            parallel.synchronize(params.values())
            torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0, error_if_nonfinite=True)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                step += 1
            report(f'Epoch {epoch + 1}/{epochs} · step {step} · loss {total / len(batches):.4f} · GPU {parallel.world()}',
                   progress=(epoch + min(1, (offset + accumulation * batch_size * parallel.world()) / len(training))) / epochs,
                   loss=total / len(batches), step=step)
            if step and step % 25 == 0:
                save(completed_epochs + epoch)
        save(completed_epochs + epoch + 1)
        if holdout and parallel.rank() == 0:
            model.eval()
            with torch.no_grad():
                validation = sum(loss_for([i]).item() for i in holdout) / len(holdout)
            report(f'Validation after epoch {epoch + 1}: {validation:.4f}', validation_loss=validation)
            if math.isfinite(validation) and validation < best_loss:
                best_loss = validation
                save(completed_epochs + epoch + 1)
                temp_best = DATA / 'models' / 'best.tmp'
                shutil.copyfile(checkpoint, temp_best)
                os.replace(temp_best, DATA / 'models' / 'best.pt')
            model.train()
        parallel.barrier()
    report('Model saved. Generate speech or continue training later.', progress=1)


def generate(report, stopped, text='', **_):
    gpu()
    checkpoint = DATA / 'models' / 'latest.pt'
    if not checkpoint.exists():
        raise ValueError('Train a model first.')
    report('Loading saved model...')
    tts = load_model()
    params = adapters(tts.model)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    for name, value in saved['adapter'].items():
        params[name].data.copy_(value)
    tts.model.eval()
    tts.voices['local'] = {}
    ref = tts._build_reference(saved['embedding'].numpy(), [], saved['reference_text'],
                               str(DATA / 'models' / 'reference.wav'))
    tts._reference_cache[('voice', 'local')] = ref
    del saved
    if stopped():
        return
    report('Generating speech on GPU...')
    audio = tts.generate(text, voice='local', use_reference_prompt=False, max_new_tokens=1000)
    if audio is None or len(audio) == 0 or not np.isfinite(audio).all():
        raise RuntimeError('The model did not generate valid audio.')
    import uuid
    name = f'{uuid.uuid4().hex[:12]}.wav'
    sf.write(DATA / 'outputs' / name, audio, tts.sample_rate)
    report('Audio is ready.', progress=1, output=name)
