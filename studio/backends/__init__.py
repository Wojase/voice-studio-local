"""TTS backend interface.

Each backend separates: availability/dependency checks, dataset preparation, training/resume,
checkpoint discovery/validation, generation and export. Capabilities that a backend does not
implement are reported as unavailable with a reason; the panel hides their controls.
"""


class Backend:
    id = ''
    label = ''
    description = ''

    def availability(self):
        """{'available': bool, 'reason': str} for inference."""
        raise NotImplementedError

    def training(self):
        """{'available': bool, 'reason': str}: real weight updates with saved, reloadable checkpoints."""
        return {'available': False, 'reason': 'Not implemented for this backend.'}

    def controls(self):
        """Generation controls this backend actually applies."""
        return {}

    def prepare_dataset(self, records):
        raise NotImplementedError

    def checkpoints(self, experiment):
        return dict(experiment.get('checkpoints', {}))

    def describe(self):
        return {'id': self.id, 'label': self.label, 'description': self.description,
                'inference': self.availability(), 'training': self.training(), 'controls': self.controls()}


class KittenBackend(Backend):
    id = 'kittentts2-lora'
    label = 'KittenTTS2 + LoRA fine-tuning'
    description = ('KittenML/kitten-tts-2 (pinned revision) with trainable low-rank adapters. Training changes adapter '
                   'weights with a cross-entropy loss on audio tokens and saves reusable checkpoints. '
                   'Outputs 24 kHz. Pretrained base voices are available for comparison.')

    def availability(self):
        try:
            import kittenml  # noqa: F401
            import torch
            if not torch.cuda.is_available():
                return {'available': False, 'reason': 'CUDA-enabled PyTorch and an NVIDIA GPU are required.'}
        except ImportError as exc:
            return {'available': False, 'reason': f'Missing dependency: {exc.name}. Run the setup script.'}
        return {'available': True, 'reason': ''}

    def training(self):
        return self.availability()

    def controls(self):
        from .kitten import CONTROLS
        return CONTROLS

    def prepare_dataset(self, records):
        from ..engine import features
        return features(lambda *a, **k: None, lambda: False, records)


class ChatterboxBackend(Backend):
    id = 'chatterbox'
    label = 'Chatterbox (pretrained, zero-shot reference cloning)'

    @property
    def description(self):
        from .chatterbox import DESCRIPTION
        return DESCRIPTION

    def availability(self):
        from .chatterbox import availability
        return availability()

    def training(self):
        from .chatterbox import TRAINING_UNAVAILABLE
        return {'available': False, 'reason': TRAINING_UNAVAILABLE}

    def controls(self):
        from .chatterbox import CONTROLS
        return CONTROLS


BACKENDS = {b.id: b for b in (KittenBackend(), ChatterboxBackend())}


def get(backend_id):
    if backend_id not in BACKENDS:
        raise ValueError('Unknown backend.')
    return BACKENDS[backend_id]
