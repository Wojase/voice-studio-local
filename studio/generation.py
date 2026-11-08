"""Generation, A/B comparison and the fixed English evaluation set, with a persistent history."""
import uuid

import soundfile as sf

from . import storage, registry
from .backends import get as backend
from .storage import read, write, now

EVALUATION_SENTENCES = [
    'The quick brown fox jumps over the lazy dog.',
    'Please call Stella and ask her to bring these things with her from the store.',
    'I honestly did not expect the meeting to run this late, but here we are.',
    'Wait, are you serious? That is the best news I have heard all week!',
    'Turn left at the second traffic light, then continue for about three hundred meters.',
    'On Tuesday, the fourteenth of March, we recorded twenty-seven new samples.',
    'She whispered, so quietly that nobody else could hear: "Do not open that door."',
    'Thank you for calling. Your appointment is confirmed for nine thirty tomorrow morning.',
    'Photosynthesis converts light energy into chemical energy stored in glucose.',
    'Well... I suppose we could try again, if you really think it will help.',
]


def validate_variant(value):
    """Normalize one variant; only controls implemented by the chosen backend are kept."""
    if not isinstance(value, dict):
        raise ValueError('Invalid generation settings.')
    b = backend(value.get('backend', 'kittentts2-lora'))
    status = b.availability()
    if not status['available']:
        raise ValueError(f'{b.label} is unavailable: {status["reason"]}')
    allowed = {'backend', 'label', 'kind', 'experiment_id', 'checkpoint', 'voice', 'reference_clip'} | set(b.controls())
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f'{b.label} does not support: ' + ', '.join(unknown))
    variant = {'backend': b.id, 'label': str(value.get('label', ''))[:20]}
    if b.id == 'kittentts2-lora':
        kind = value.get('kind', 'finetuned')
        if kind not in ('finetuned', 'base'):
            raise ValueError('Choose a fine-tuned experiment or the pretrained base.')
        variant['kind'] = kind
        if kind == 'finetuned':
            experiment = registry.load(str(value.get('experiment_id', '')))
            which = str(value.get('checkpoint') or ('best' if experiment.get('best') else 'latest'))
            registry.checkpoint_path(experiment, which)  # Validates BEST / LATEST / snapshot name now.
            variant.update(experiment_id=experiment['id'], checkpoint=which)
        else:
            variant['voice'] = str(value.get('voice') or '') or None
    else:
        variant['kind'] = 'base'
    if value.get('reference_clip'):
        from .management import safe_key
        clip = safe_key(str(value['reference_clip']))
        if not (storage.DATA / 'clips' / (clip + '.wav')).is_file():
            raise ValueError('Reference clip not found.')
        if b.id == 'chatterbox' and sf.info(storage.DATA / 'clips' / (clip + '.wav')).duration < 5:
            raise ValueError('Chatterbox reference clips should be at least 5 seconds long.')
        variant['reference_clip'] = clip
    for name, spec in b.controls().items():
        raw = value.get(name)
        if raw is None or raw == '':
            continue
        if spec['type'] == 'choice':
            if raw not in spec['choices']:
                raise ValueError(f'Invalid {name}.')
            variant[name] = raw
            continue
        number = int(raw) if spec['type'] == 'int' else float(raw)
        if not spec['min'] <= number <= spec['max']:
            raise ValueError(f"{name} must be between {spec['min']} and {spec['max']}.")
        variant[name] = number
    return variant


def history(limit=200):
    rows = [read(p) for p in (storage.DATA / 'outputs').glob('*.json')]
    rows = [r for r in rows if r and (storage.DATA / 'outputs' / r['file']).exists()]
    return sorted(rows, key=lambda r: r.get('created', ''), reverse=True)[:limit]


def record(variant, info, text, file, sample_rate, seconds, group, index):
    meta = {'id': file.rsplit('.', 1)[0], 'file': file, 'created': now(), 'group': group, 'index': index,
            'backend': variant['backend'], 'label': variant.get('label', ''), 'kind': info['kind'],
            'trained': info['kind'] == 'finetuned', 'text': text, 'sample_rate': sample_rate, 'seconds': round(seconds, 2),
            'settings': {k: v for k, v in variant.items() if k not in ('backend', 'label', 'kind')}, **info}
    write(storage.DATA / 'outputs' / (meta['id'] + '.json'), meta)
    return meta


def run(report, stopped, variants=None, texts=None, text='', group=None, **_):
    """Every variant speaks every text (A/B uses identical text). Variants are loaded one at a time."""
    if not variants:  # Backward compatibility: plain {"text"} uses the newest experiment's best/latest.
        variants = [{'backend': 'kittentts2-lora', 'kind': 'finetuned', 'experiment_id': registry.listing()[0]['id'],
                     'checkpoint': 'latest'}]
    texts = texts or [text]
    group = group or uuid.uuid4().hex[:10]
    total, done, outputs = len(variants) * len(texts), 0, []
    for index, variant in enumerate(variants):
        if stopped():
            return
        label = variant.get('label') or chr(65 + index)
        if variant['backend'] == 'chatterbox':
            from .backends import chatterbox
            report(f'{label}: loading Chatterbox in its own environment...', progress=done / total)
            reference = str(storage.DATA / 'clips' / (variant['reference_clip'] + '.wav')) if variant.get('reference_clip') else None
            controls = {k: variant[k] for k in chatterbox.CONTROLS if k in variant}
            items = [dict(controls, text=t, reference=reference,
                          out=str(storage.DATA / 'outputs' / f'{uuid.uuid4().hex[:12]}.wav')) for t in texts]
            result = chatterbox.generate(items)
            info = {'kind': 'base', 'model': chatterbox.MODEL, 'revision': chatterbox.REVISION,
                    'reference_clip': variant.get('reference_clip'), 'voice': None if reference else 'built-in default',
                    'watermarked': True}
            for t, item, produced in zip(texts, items, result['items']):
                name = item['out'].replace('\\', '/').rsplit('/', 1)[-1]
                outputs.append(record(variant, info, t, name, chatterbox.SAMPLE_RATE, produced['seconds'], group, index))
                done += 1
            continue
        from .backends.kitten import Synthesizer
        report(f'{label}: loading KittenTTS2...', progress=done / total)
        synth = Synthesizer(variant)
        try:
            for t in texts:
                if stopped():
                    return
                report(f'{label}: generating {done + 1}/{total}...', progress=done / total)
                audio, rate = synth(t)
                name = f'{uuid.uuid4().hex[:12]}.wav'
                sf.write(storage.DATA / 'outputs' / name, audio, rate)
                outputs.append(record(variant, synth.info, t, name, rate, len(audio) / rate, group, index))
                done += 1
        finally:
            synth.close()
    report('Audio is ready.', progress=1, output=outputs[0]['file'] if outputs else None, group=group)
