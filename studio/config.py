"""Training settings for the KittenTTS2 LoRA backend.

FIXED settings define the adapter or the experiment's identity and can only be chosen when an
experiment is created (duplicate the experiment to change them). RESUMABLE settings apply to
one training run and may change every time an experiment is resumed.
"""
LR_PRESETS = {'very_gentle': 1e-5, 'gentle': 5e-5, 'standard': 1e-4, 'strong': 2e-4}
TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
SCHEDULERS = ('constant', 'cosine', 'linear')

# name: (type, default, min, max, help)
FIXED = {
    'rank': (int, 8, 1, 128, 'LoRA rank: adapter capacity. Changes the shape of the saved weights.'),
    'alpha': (float, 16.0, .1, 1024, 'LoRA alpha. The adapter output is scaled by alpha / rank.'),
    'dropout': (float, 0.0, 0.0, .5, 'Dropout applied to the adapter input during training only.'),
    'targets': (list, ['q_proj', 'v_proj'], None, None, 'Linear layers in each of the 28 transformer blocks that receive adapters.'),
    'seed': (int, 42, 0, 2 ** 31 - 1, 'Random seed for adapter initialization, data order and the validation split.'),
    'validation_fraction': (float, .1, 0.0, .5, 'Share of original recordings held out for validation (0 disables validation).'),
}
RESUMABLE = {
    'lr': (float, 5e-5, 1e-6, 1e-3, 'Peak learning rate for this run.'),
    'epochs': (int, 3, 1, 1000, 'Additional passes over the training clips in this run.'),
    'max_updates': (int, 0, 0, 1_000_000, 'Stop this run after this many optimizer updates (0 = no limit).'),
    'batch_size': (int, 1, 1, 32, 'Clips processed at once on each GPU. Limited by that card\'s own memory.'),
    'accumulation': (int, 2, 1, 64, 'Micro-batches summed before each optimizer update.'),
    'weight_decay': (float, 0.0, 0.0, .5, 'AdamW decoupled weight decay on adapter weights.'),
    'grad_clip': (float, 1.0, 0.0, 100.0, 'Clip the global gradient norm to this value (0 disables clipping).'),
    'warmup_updates': (int, 0, 0, 100_000, 'Linear warmup from 0 to the peak learning rate over the first updates of this run.'),
    'scheduler': (str, 'constant', None, None, 'Learning-rate schedule over this run after warmup: constant, cosine (to 10%) or linear (to 0).'),
    'checkpointing': (bool, True, None, None, 'Activation checkpointing: recompute activations to save GPU memory (slower).'),
    'save_every': (int, 25, 1, 100_000, 'Save latest.pt every N updates (also saved after every epoch and on stop).'),
    'keep_checkpoints': (int, 3, 0, 100, 'Number of numbered snapshots kept; best.pt and latest.pt are always kept.'),
    'validate_every': (int, 0, 0, 100_000, 'Validate every N updates (0 = only after each epoch).'),
    'early_stop_patience': (int, 0, 0, 1000, 'Stop after this many validations without improvement (0 disables).'),
    'early_stop_min_delta': (float, 0.0, 0.0, 10.0, 'Minimum validation-loss decrease that counts as an improvement.'),
    'gpus': (list, [0], None, None, 'GPU indices to train on. Two or more use one process per GPU (Linux + NCCL only).'),
}
LEGACY_ARCHITECTURE = {'rank': 8, 'alpha': 16.0, 'dropout': 0.0, 'targets': ['q_proj', 'v_proj']}


def _coerce(name, spec, value):
    kind, default, low, high, _ = spec
    if value is None or value == '':
        return list(default) if isinstance(default, list) else default
    if kind is bool:
        return value if isinstance(value, bool) else str(value).lower() in ('1', 'true', 'yes', 'on')
    if kind is list:
        if not isinstance(value, list):
            raise ValueError(f'{name} must be a list.')
        return list(value)
    try:
        value = kind(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be a {kind.__name__}.')
    if low is not None and not low <= value <= high:
        raise ValueError(f'{name} must be between {low} and {high}.')
    return value


def validate(values, gpu_count=None, distributed_ok=True):
    values = values or {}
    unknown = set(values) - set(FIXED) - set(RESUMABLE) - {'lr_preset'}
    if unknown:
        raise ValueError('Unknown training settings: ' + ', '.join(sorted(unknown)))
    result = {name: _coerce(name, spec, values.get(name)) for name, spec in {**FIXED, **RESUMABLE}.items()}
    if values.get('lr_preset') and values.get('lr') in (None, ''):
        if values['lr_preset'] not in LR_PRESETS:
            raise ValueError('Unknown learning-rate preset.')
        result['lr'] = LR_PRESETS[values['lr_preset']]
    targets = result['targets']
    if not targets or any(t not in TARGETS for t in targets) or len(set(targets)) != len(targets):
        raise ValueError('LoRA targets must be a non-empty subset of: ' + ', '.join(TARGETS))
    result['targets'] = [t for t in TARGETS if t in targets]  # Canonical order.
    if result['scheduler'] not in SCHEDULERS:
        raise ValueError('Scheduler must be constant, cosine or linear.')
    gpus = result['gpus']
    if not gpus or len(set(gpus)) != len(gpus) or any(not isinstance(g, int) or g < 0 for g in gpus):
        raise ValueError('Choose one or more distinct GPU indices.')
    if gpu_count is not None and any(g >= gpu_count for g in gpus):
        raise ValueError(f'Only {gpu_count} GPU(s) detected.')
    if len(gpus) > 1 and not distributed_ok:
        raise ValueError('Multi-GPU training requires Linux with NCCL. Select one GPU on this system.')
    return result


def architecture(config):
    return {key: config[key] for key in ('rank', 'alpha', 'dropout', 'targets')}


def check_resume(experiment_config, new_config):
    """Refuse silent changes to fixed settings of an experiment that already has weights."""
    changed = [k for k in FIXED if experiment_config.get(k) != new_config.get(k)]
    if changed:
        raise ValueError('These settings are fixed for an existing experiment: ' + ', '.join(changed)
                         + '. Duplicate the experiment to train with different values.')


def schema():
    def describe(group):
        return {name: {'type': spec[0].__name__, 'default': spec[1], 'min': spec[2], 'max': spec[3], 'help': spec[4]}
                for name, spec in group.items()}
    return {'fixed': describe(FIXED), 'resumable': describe(RESUMABLE), 'lr_presets': LR_PRESETS,
            'targets': list(TARGETS), 'schedulers': list(SCHEDULERS)}


def learning_rate(peak, update, total, warmup, scheduler):
    """Learning rate for this run's `update` (0-based) out of `total` planned updates."""
    if warmup and update < warmup:
        return peak * (update + 1) / warmup
    if scheduler == 'constant' or not total:
        return peak
    progress = min(1.0, (update - warmup) / max(1, total - warmup))
    if scheduler == 'linear':
        return peak * max(0.0, 1 - progress)
    import math
    return peak * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
