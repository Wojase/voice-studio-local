"""KittenTTS2 + LoRA backend: real fine-tuning (cross-entropy on target audio tokens, AdamW updates)."""
import gc
import json
import math
import os
import random
import shutil
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .. import storage, registry, config as settings, datasets
from .. import distributed as parallel
from ..storage import read, write, append_jsonl, now, sample_of

FORMAT = 'voice-studio/kittentts2-lora/2'
SAMPLE_RATE = 24000
MAX_TOKENS = 1024


def layer_shapes():
    """(in, out) features of each LoRA-capable layer, from the pinned base model's own config."""
    from ..engine import repository
    path, _ = repository()
    lm = json.loads((Path(path) / 'lm' / 'config.json').read_text(encoding='utf-8'))
    hidden, inter, head = lm['hidden_size'], lm['intermediate_size'], lm.get('head_dim', lm['hidden_size'] // lm['num_attention_heads'])
    q, kv = lm['num_attention_heads'] * head, lm['num_key_value_heads'] * head
    shapes = {'q_proj': (hidden, q), 'k_proj': (hidden, kv), 'v_proj': (hidden, kv), 'o_proj': (q, hidden),
              'gate_proj': (hidden, inter), 'up_proj': (hidden, inter), 'down_proj': (inter, hidden)}
    group = {'q_proj': 'self_attn', 'k_proj': 'self_attn', 'v_proj': 'self_attn', 'o_proj': 'self_attn',
             'gate_proj': 'mlp', 'up_proj': 'mlp', 'down_proj': 'mlp'}
    return lm['num_hidden_layers'], shapes, group


def expected_adapter(architecture):
    """Exact tensor names and shapes an adapter with this architecture must contain."""
    layers, shapes, group = layer_shapes()
    rank = architecture['rank']
    expected = {}
    for i in range(layers):
        for target in architecture['targets']:
            fan_in, fan_out = shapes[target]
            prefix = f'model.layers.{i}.{group[target]}.{target}'
            expected[prefix + '.a'] = (rank, fan_in)
            expected[prefix + '.b'] = (fan_out, rank)
    return expected


def validate_checkpoint(saved, experiment, expected=None):
    """Refuse weights whose backend, base model or adapter shape differ from the experiment."""
    from ..engine import BASE, REVISION
    if saved.get('backend', registry.KITTEN) != registry.KITTEN:
        raise ValueError('Checkpoint belongs to a different backend.')
    if saved.get('base') != BASE or saved.get('revision') != REVISION or saved.get('weights', 'emb4') != 'emb4':
        raise ValueError('Checkpoint base model or revision does not match this backend.')
    architecture = saved.get('architecture') or settings.LEGACY_ARCHITECTURE
    if architecture != experiment['architecture']:
        raise ValueError('Checkpoint adapter architecture differs from this experiment; refusing to load it.')
    adapter = saved.get('adapter') or {}
    expected = expected or expected_adapter(architecture)
    if set(adapter) != set(expected):
        raise ValueError('Checkpoint adapter tensors do not match the expected layers.')
    for name, shape in expected.items():
        if tuple(adapter[name].shape) != tuple(shape) or not torch.isfinite(adapter[name]).all():
            raise ValueError(f'Adapter tensor {name} has an invalid shape or values.')
    if saved.get('embedding') is None or tuple(saved['embedding'].shape) != (512,):
        raise ValueError('Checkpoint lacks a valid speaker embedding.')
    return architecture


def load_checkpoint(path):
    # weights_only=True: restricted unpickler (tensors and plain containers only).
    return torch.load(path, map_location='cpu', weights_only=True)


# --- split -------------------------------------------------------------------------------------

def bucket(key, seed):
    return zlib.crc32(f'{seed}:{key}'.encode()) % 1000


def build_split(path, records, cfg):
    """Validation split by ORIGINAL RECORDING, persisted so membership never moves.

    Legacy experiments keep their clip-level split (explicitly marked). New recordings added later are
    assigned deterministically; they never move existing members between partitions.
    """
    split = read(path)
    fraction = cfg['validation_fraction']
    if split and split.get('kind') == 'clip-legacy':
        for row in records:
            if row['id'] not in split['clips']:
                split['clips'][row['id']] = 'validation' if bucket(row['id'], 0) < 100 else 'train'
        parts = {row['id']: split['clips'][row['id']] for row in records}
        return split, parts
    if not split:
        recordings = sorted({sample_of(r) for r in records}, key=lambda k: bucket(k, cfg['seed']))
        counts = {k: sum(1 for r in records if sample_of(r) == k) for k in recordings}
        membership = {k: 'train' for k in recordings}
        if fraction > 0 and len(recordings) >= 2:
            target, held = fraction * len(records), 0
            # Prefer smaller recordings so most data stays in training; always keep one for training.
            for key in sorted(recordings, key=lambda k: (counts[k], bucket(k, cfg['seed']))):
                if held >= target or sum(1 for v in membership.values() if v == 'train') <= 1:
                    break
                membership[key] = 'validation'
                held += counts[key]
        split = {'kind': 'recording', 'fraction': fraction, 'seed': cfg['seed'], 'created': now(), 'recordings': membership}
    for key in {sample_of(r) for r in records}:
        if key not in split['recordings']:
            split['recordings'][key] = 'validation' if fraction and bucket(key, cfg['seed']) < fraction * 1000 else 'train'
    parts = {row['id']: split['recordings'][sample_of(row)] for row in records}
    return split, parts


def split_warning(validation_rows, kind):
    seconds = sum(r.get('seconds', 0) for r in validation_rows)
    notes = []
    if kind == 'clip-legacy':
        notes.append('Legacy clip-level split: validation clips may come from training recordings, so validation loss is optimistic.')
    if not validation_rows:
        notes.append('No validation set (needs at least two recordings for a recording-level split). BEST cannot be selected.')
    elif len(validation_rows) < 5 or seconds < 60:
        notes.append(f'Tiny validation set ({len(validation_rows)} clips, {seconds:.0f} s): validation loss is noisy. Rely on listening tests.')
    return notes


# --- training ----------------------------------------------------------------------------------

def train(report, stopped, experiment_id, run=None, **_):
    from ..engine import gpu, load_model, features, BASE, REVISION
    gpu()
    experiment = registry.load(experiment_id)
    exp_dir = registry.root() / experiment_id
    cfg = settings.validate({**{k: v for k, v in experiment['config'].items() if k in settings.FIXED}, **(run or {})})
    settings.check_resume(experiment['config'], cfg)
    if not experiment.get('resumable', True):
        raise ValueError('This experiment cannot be resumed (imported weights have no optimizer state). Duplicate it to train.')
    seed = cfg['seed']
    lead = parallel.rank() == 0

    def note(message, **fields):
        report(message, experiment_id=experiment_id, **fields)
        if lead:
            registry.log(experiment_id, message)

    records = sorted(datasets.training_clips(experiment['dataset_id'], experiment['speaker']), key=lambda r: r['id'])
    if len(records) < 2:
        raise ValueError('At least two enabled clips from this dataset and speaker are required.')
    if lead:
        features(report, stopped, records)
    parallel.barrier()
    if parallel.should_stop(stopped):
        return
    split, parts = build_split(exp_dir / 'split.json', records, cfg)
    training = [i for i, r in enumerate(records) if parts[r['id']] == 'train']
    holdout = [i for i, r in enumerate(records) if parts[r['id']] == 'validation']
    if not training:
        raise ValueError('Every enabled clip is in the validation split. Add recordings or lower the validation fraction.')
    if lead:
        write(exp_dir / 'split.json', split)
        for warning in split_warning([records[i] for i in holdout], split['kind']):
            note('Warning: ' + warning)

    note('Loading training model...')
    tts = load_model(training=True)
    model = tts.model
    architecture = settings.architecture(cfg)
    torch.manual_seed(seed)  # Identical adapter initialization on every rank.
    params = adapters_for(model, architecture)
    optimizer = torch.optim.AdamW(list(params.values()), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    step, completed_epochs = experiment.get('updates', 0), experiment.get('epochs', 0)
    best_loss = experiment.get('best_validation_loss')
    best_loss = float('inf') if best_loss is None else best_loss
    if experiment.get('latest'):
        saved = load_checkpoint(registry.checkpoint_path(experiment, 'latest'))
        validate_checkpoint(saved, experiment, {k: tuple(v.shape) for k, v in params.items()})
        for name, value in saved['adapter'].items():
            params[name].data.copy_(value)
        if 'optimizer' not in saved:
            raise ValueError('The latest checkpoint has no optimizer state and cannot be resumed.')
        optimizer.load_state_dict(saved['optimizer'])
        step, completed_epochs = saved['step'], saved['epochs']
        note(f'Resuming from latest.pt at update {step} ({completed_epochs} completed epochs).')
        del saved
    if parallel.world() > 1:
        for parameter in params.values():
            torch.distributed.broadcast(parameter.data, src=0)
    for group in optimizer.param_groups:
        group['weight_decay'] = cfg['weight_decay']

    from kittenml.kittentts2.prompt import build_generation_prompt
    from kittenml.kittentts2.tokens import TokenMap
    tm = TokenMap.from_config(tts.config)
    examples = []
    for row in records:
        feature = torch.load(storage.DATA / 'features' / (row['id'] + '.pt'), weights_only=True)
        prompt = build_generation_prompt(tm, tts.tokenizer.encode(row['text'], add_special_tokens=False))
        target = [tm.audio_id_base + t for t in feature['tokens']] + [tm.speech_end_id]
        if len(prompt) + len(target) > MAX_TOKENS:
            raise ValueError(f"Clip is too long: {row['id']}. Shorten the transcript or disable this clip.")
        examples.append((prompt, target, feature['embedding']))
    voice_embedding = nn.functional.normalize(torch.stack([examples[i][2] for i in training]).mean(0), dim=0)
    reference = experiment.get('reference')
    if not reference or not (exp_dir / 'reference.wav').is_file():
        first = records[training[0]]
        if lead:
            shutil.copyfile(storage.DATA / 'clips' / (first['id'] + '.wav'), exp_dir / 'reference.wav')
        reference = {'clip_id': first['id'], 'text': first['text']}
    if experiment.get('legacy') and experiment.get('latest'):
        # Keep the legacy reference transcript that matches the legacy reference.wav.
        legacy = load_checkpoint(registry.checkpoint_path(experiment, 'latest'))
        reference = {'clip_id': None, 'text': legacy.get('reference_text', reference['text'])}
        voice_embedding = legacy['embedding']

    if cfg['checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    model.train()

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
        with torch.autocast('cuda', dtype=torch.bfloat16):
            inputs = model.get_input_embeddings()(ids)
            embeddings_batch = torch.stack([e for _, _, e in selected]).to('cuda', dtype=torch.bfloat16)
            inputs = torch.cat([model.spk_proj(embeddings_batch).unsqueeze(1), inputs[:, 1:]], dim=1)
            hidden = model.model(inputs_embeds=inputs, attention_mask=mask, use_cache=False).last_hidden_state
            valid = labels != -100
            logits = model.lm_head(hidden[valid])  # Only target audio positions.
            return nn.functional.cross_entropy(logits.float(), labels[valid])

    def validate():
        """Validation spread over all ranks, summed with all_reduce."""
        if not holdout:
            return None
        model.eval()
        total = torch.zeros(2, device='cuda', dtype=torch.float64)
        with torch.no_grad():
            for i in holdout[parallel.rank()::parallel.world()]:
                total[0] += loss_for([i]).item()
                total[1] += 1
        if parallel.world() > 1:
            torch.distributed.all_reduce(total)
        model.train()
        return (total[0] / total[1]).item()

    world = parallel.world()
    per_update = cfg['accumulation'] * cfg['batch_size'] * world
    updates_per_epoch = math.ceil(len(training) / per_update)
    planned = cfg['epochs'] * updates_per_epoch
    if cfg['max_updates']:
        planned = min(planned, cfg['max_updates'])
    run_record = {'started': now(), 'start_update': step, 'config': cfg, 'gpus': world,
                  'train_clips': len(training), 'validation_clips': len(holdout), 'planned_updates': planned,
                  'updates_per_epoch': updates_per_epoch, 'status': 'running'}
    if lead:
        experiment.update(status='training', config=cfg, reference=reference)
        experiment.setdefault('runs', []).append(run_record)
        registry.save(experiment)
    note(f'Training {len(training)} clips, validating {len(holdout)} · {world} GPU(s) · {per_update} clips per update · '
         f'{updates_per_epoch} updates/epoch · {planned} planned updates.')

    state = {'last_validation': None, 'bad': 0}

    def save(epochs_done, validation=None, snapshot=False, best=False):
        if not lead:
            return
        payload = {'format': FORMAT, 'backend': registry.KITTEN, 'base': BASE, 'revision': REVISION, 'weights': 'emb4',
                   'architecture': architecture, 'adapter': {k: v.detach().cpu() for k, v in params.items()},
                   'optimizer': optimizer.state_dict(), 'step': step, 'epochs': epochs_done,
                   'best_loss': best_loss, 'validation_loss': validation, 'embedding': voice_embedding.cpu(),
                   'reference_text': reference['text'], 'experiment_id': experiment_id, 'config': cfg, 'saved': now()}
        folder = exp_dir / 'checkpoints'
        temp = folder / 'latest.tmp'
        torch.save(payload, temp)
        os.replace(temp, folder / 'latest.pt')
        info = {'update': step, 'epochs': epochs_done, 'validation_loss': validation, 'saved': now()}
        registry.record_checkpoint(experiment, 'latest.pt', info)
        experiment['latest'] = {'name': 'latest.pt', 'update': step}
        if snapshot and cfg['keep_checkpoints']:
            name = f'update-{step:06d}.pt'
            shutil.copyfile(folder / 'latest.pt', folder / name)
            registry.record_checkpoint(experiment, name, dict(info))
            registry.prune(experiment, cfg['keep_checkpoints'])
        if best:
            shutil.copyfile(folder / 'latest.pt', folder / 'best.tmp')
            os.replace(folder / 'best.tmp', folder / 'best.pt')
            registry.record_checkpoint(experiment, 'best.pt', dict(info))
            experiment['best'] = {'name': 'best.pt', 'update': step, 'validation_loss': validation}
            experiment['best_validation_loss'] = validation
        experiment.update(updates=step, epochs=epochs_done)
        registry.save(experiment)

    def run_validation(epochs_done):
        nonlocal best_loss
        validation = validate()
        if validation is None:
            return False
        state['last_validation'] = validation
        improved = math.isfinite(validation) and validation < best_loss
        meaningful = validation < best_loss - cfg['early_stop_min_delta']
        state['bad'] = 0 if meaningful else state['bad'] + 1
        if improved:
            best_loss = validation
        if lead:
            append_jsonl(exp_dir / 'metrics.jsonl', {'t': time.time(), 'update': step, 'validation_loss': validation})
        save(epochs_done, validation, best=improved)
        note(f'Validation at update {step}: {validation:.4f}' + (' · new best' if improved else ''), validation_loss=validation)
        return bool(cfg['early_stop_patience'] and state['bad'] >= cfg['early_stop_patience'])

    def finish(status, message):
        if lead:
            run_record.update(finished=now(), end_update=step, status=status)
            experiment['status'] = 'ready' if status in ('done', 'stopped', 'early_stopped', 'limit') else status
            registry.save(experiment)
        note(message, progress=1 if status != 'stopped' else None)

    started, run_updates, stop_reason = time.time(), 0, None
    randomizer = random.Random(seed + step)
    try:
        for epoch in range(cfg['epochs']):
            order = list(training)
            randomizer.shuffle(order)
            for offset, batches in parallel.groups(order, cfg['accumulation'], cfg['batch_size'], parallel.rank(), world):
                if parallel.should_stop(stopped):
                    save(completed_epochs, state['last_validation'])
                    finish('stopped', f'Stopped and saved at update {step}. Resume continues from latest.pt (the interrupted epoch restarts).')
                    return
                lr = settings.learning_rate(cfg['lr'], run_updates, planned, cfg['warmup_updates'], cfg['scheduler'])
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                total = 0.
                for batch in batches:
                    loss = loss_for(batch)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Training became unstable (non-finite loss). The last saved checkpoint is preserved.')
                    total += loss.item()
                    (loss / len(batches)).backward()
                parallel.synchronize(params.values())
                if cfg['grad_clip']:
                    torch.nn.utils.clip_grad_norm_(list(params.values()), cfg['grad_clip'], error_if_nonfinite=True)
                optimizer.step()
                step += 1
                run_updates += 1
                loss_value = total / len(batches)
                if world > 1:
                    value = torch.tensor(loss_value, device='cuda')
                    torch.distributed.all_reduce(value)
                    loss_value = value.item() / world
                elapsed = time.time() - started
                rate = run_updates / elapsed
                eta = (planned - run_updates) / rate if run_updates >= 3 and rate > 0 else None
                if lead:
                    append_jsonl(exp_dir / 'metrics.jsonl', {'t': time.time(), 'update': step, 'epoch': completed_epochs + epoch + 1,
                                                             'loss': loss_value, 'lr': lr})
                report(f'Epoch {epoch + 1}/{cfg["epochs"]} · update {step} · loss {loss_value:.4f} · {world} GPU(s)',
                       progress=run_updates / planned, loss=loss_value, step=step, lr=lr, elapsed=round(elapsed),
                       eta=round(eta) if eta is not None else None, updates_per_second=round(rate, 3),
                       clips_per_second=round(rate * per_update, 2), experiment_id=experiment_id)
                if cfg['validate_every'] and step % cfg['validate_every'] == 0 and run_validation(completed_epochs + epoch):
                    stop_reason = 'early_stopped'
                    break
                if step % cfg['save_every'] == 0:
                    save(completed_epochs + epoch, state['last_validation'], snapshot=True)
                if run_updates >= planned:
                    stop_reason = 'limit' if run_updates < cfg['epochs'] * updates_per_epoch else None
                    break
            if stop_reason:
                break
            completed_epochs += 1
            save(completed_epochs, state['last_validation'], snapshot=True)
            if run_validation(completed_epochs):
                stop_reason = 'early_stopped'
                break
            parallel.barrier()
        save(completed_epochs, state['last_validation'])
        if stop_reason == 'early_stopped':
            finish('early_stopped', f'Early stopping: no validation improvement for {cfg["early_stop_patience"]} checks. Saved update {step}.')
        elif stop_reason == 'limit':
            finish('limit', f'Reached the update limit ({cfg["max_updates"]}). Saved update {step}; the partial epoch is not counted.')
        else:
            finish('done', f'Training run complete at update {step}. Compare BEST and LATEST by listening.')
    except Exception:
        if lead:
            run_record.update(finished=now(), end_update=step, status='error')
            experiment['status'] = 'error'
            registry.save(experiment)
        raise
    finally:
        del model, tts
        gc.collect()
        torch.cuda.empty_cache()


def adapters_for(model, architecture):
    from ..engine import adapters
    return adapters(model, architecture['rank'], architecture['alpha'], architecture['dropout'], tuple(architecture['targets']))


# --- inference ---------------------------------------------------------------------------------

class Synthesizer:
    """One loaded KittenTTS2 model: either the pretrained base or base + a validated adapter."""

    def __init__(self, variant):
        from ..engine import gpu, load_model
        gpu()
        self.variant = variant
        self.tts = load_model()
        self.info = {'kind': variant['kind']}
        if variant['kind'] == 'finetuned':
            experiment = registry.load(variant['experiment_id'])
            path = registry.checkpoint_path(experiment, variant.get('checkpoint', 'best'))
            saved = load_checkpoint(path)
            architecture = validate_checkpoint(saved, experiment)
            params = adapters_for(self.tts.model, architecture)
            for name, value in saved['adapter'].items():
                params[name].data.copy_(value)
            self.tts.model.eval()
            reference_wav = registry.root() / experiment['id'] / 'reference.wav'
            reference_text = saved['reference_text']
            if variant.get('reference_clip'):
                reference_wav, reference_text = clip_reference(variant['reference_clip'])
            elif not reference_wav.is_file():
                raise ValueError('This experiment has no reference.wav (it was not included in the import). Choose a reference clip.')
            self.tts._reference_cache[('voice', 'local')] = self.tts._build_reference(
                saved['embedding'].numpy(), [], reference_text, str(reference_wav))
            self.tts.voices['local'] = {}
            self.voice = 'local'
            self.info.update(experiment_id=experiment['id'], experiment_name=experiment['name'],
                             checkpoint=path.name, requested_checkpoint=variant.get('checkpoint', 'best'),
                             update=saved['step'], validation_loss=saved.get('validation_loss'))
        else:
            self.tts.model.eval()
            self.voice = None
            if variant.get('reference_clip'):
                wav, text = clip_reference(variant['reference_clip'])
                self.reference = (str(wav), text)
                self.info['reference_clip'] = variant['reference_clip']
            else:
                self.voice = variant.get('voice') or self.tts.config.get('default_voice')
                if self.voice not in self.tts.voices:
                    raise ValueError('Unknown built-in voice.')
                self.info['voice'] = self.voice

    def __call__(self, text):
        v = self.variant
        kwargs = dict(preset=v.get('preset') or None, temperature=v.get('temperature'), top_p=v.get('top_p'),
                      top_k=v.get('top_k'), min_p=v.get('min_p'), max_new_tokens=int(v.get('max_new_tokens') or 1000),
                      advanced={'seed': v.get('seed')} if v.get('seed') is not None else None)
        if self.voice == 'local':
            audio = self.tts.generate(text, voice='local', use_reference_prompt=False, **kwargs)
        elif self.voice:
            audio = self.tts.generate(text, voice=self.voice, **kwargs)
        else:
            audio = self.tts.generate(text, reference=self.reference[0], reference_text=self.reference[1], **kwargs)
        if audio is None or len(audio) == 0 or not np.isfinite(audio).all():
            raise RuntimeError('The model did not generate valid audio.')
        return audio, self.tts.sample_rate

    def close(self):
        del self.tts
        gc.collect()
        torch.cuda.empty_cache()


def clip_reference(clip_id):
    from ..management import safe_key
    row = read(storage.DATA / 'clips' / (safe_key(clip_id) + '.json'))
    wav = storage.DATA / 'clips' / (clip_id + '.wav')
    if not row or not wav.is_file():
        raise ValueError('Reference clip not found.')
    return wav, row['text']


def builtin_voices():
    from ..engine import repository
    path, cfg = repository()
    voices = Path(path) / cfg.get('voices', 'voices/voices.json')
    names = sorted(json.loads(voices.read_text(encoding='utf-8'))) if voices.is_file() else [cfg.get('default_voice', 'Bruno')]
    return [n for n in names if n not in NON_ENGLISH]  # This studio is English-only.


NON_ENGLISH = {'Arabic', 'Chinese', 'French', 'German', 'Hindi', 'Italian', 'Portuguese', 'Russian', 'Spanish'}


CONTROLS = {
    'preset': {'type': 'choice', 'choices': ['stable', 'expressive'], 'default': 'stable', 'help': 'Base sampling preset from the model config.'},
    'temperature': {'type': 'float', 'min': .1, 'max': 1.5, 'default': None, 'help': 'Overrides the preset. Higher is more varied, less stable.'},
    'top_p': {'type': 'float', 'min': .1, 'max': 1.0, 'default': None, 'help': 'Nucleus sampling cut-off (overrides the preset).'},
    'top_k': {'type': 'int', 'min': 0, 'max': 500, 'default': None, 'help': 'Sample only from the k most likely tokens (overrides the preset).'},
    'min_p': {'type': 'float', 'min': 0.0, 'max': .5, 'default': None, 'help': 'Discard tokens below this fraction of the top probability.'},
    'seed': {'type': 'int', 'min': 0, 'max': 2 ** 31 - 1, 'default': None, 'help': 'Fixed seed for reproducible sampling.'},
    'max_new_tokens': {'type': 'int', 'min': 50, 'max': 3000, 'default': 1000, 'help': 'Per-chunk audio token budget (25 tokens ≈ 1 second).'},
}
