"""Named datasets and speakers, per-recording assignment and clip quality warnings."""
import re
import uuid

import numpy as np
import soundfile as sf

from . import storage
from .storage import read, write, sample_of

DEFAULT_ID = 'default'


def registry_path():
    return storage.DATA / 'datasets.json'


def load():
    value = read(registry_path())
    if not value:
        value = {'version': 1, 'datasets': [{'id': DEFAULT_ID, 'name': 'Default dataset', 'speakers': ['Speaker 1']}]}
        write(registry_path(), value)
    return value


def find(dataset_id):
    for row in load()['datasets']:
        if row['id'] == dataset_id:
            return row
    raise ValueError('Dataset not found.')


def clean_name(value, label='Name'):
    value = re.sub(r'\s+', ' ', str(value or '')).strip()
    if not 1 <= len(value) <= 80:
        raise ValueError(f'{label} must contain 1-80 characters.')
    return value


def create(name, speaker):
    value = load()
    row = {'id': uuid.uuid4().hex[:10], 'name': clean_name(name), 'speakers': [clean_name(speaker, 'Speaker')]}
    value['datasets'].append(row)
    write(registry_path(), value)
    return row


def update(dataset_id, name=None, add_speaker=None):
    value = load()
    for row in value['datasets']:
        if row['id'] == dataset_id:
            if name is not None:
                row['name'] = clean_name(name)
            if add_speaker:
                speaker = clean_name(add_speaker, 'Speaker')
                if speaker not in row['speakers']:
                    row['speakers'].append(speaker)
            write(registry_path(), value)
            return row
    raise ValueError('Dataset not found.')


def assign(sample_ids, dataset_id, speaker):
    from .management import safe_key
    dataset = find(dataset_id)
    if speaker not in dataset['speakers']:
        raise ValueError('Add this speaker to the dataset first.')
    changed = []
    for key in sample_ids:
        path = storage.DATA / 'samples' / (safe_key(key) + '.json')
        row = read(path)
        if not row:
            raise ValueError(f'Recording not found: {key}')
        row.update(dataset_id=dataset_id, speaker=speaker)
        write(path, row)
        changed.append(row)
    return changed


def migrate():
    """Existing recordings without a dataset join the default dataset and its first speaker."""
    value = load()
    default = next(d for d in value['datasets'] if d['id'] == DEFAULT_ID) if any(
        d['id'] == DEFAULT_ID for d in value['datasets']) else value['datasets'][0]
    for path in (storage.DATA / 'samples').glob('*.json'):
        row = read(path)
        if row and 'dataset_id' not in row:
            row.update(dataset_id=default['id'], speaker=default['speakers'][0])
            if 'sample_rate' not in row:
                try:
                    row['sample_rate'] = sf.info(path.with_suffix('.wav')).samplerate
                except RuntimeError:
                    pass
            write(path, row)


def recordings(dataset_id=None, speaker=None):
    rows = [read(p) for p in sorted((storage.DATA / 'samples').glob('*.json'))]
    return [r for r in rows if r and (dataset_id is None or r.get('dataset_id') == dataset_id)
            and (speaker is None or r.get('speaker') == speaker)]


def training_clips(dataset_id, speaker):
    """Enabled clips whose original recording belongs to this dataset and speaker."""
    allowed = {r['id'] for r in recordings(dataset_id, speaker)}
    rows = [read(p) for p in storage.clips()]
    return [r for r in rows if r and r.get('enabled', True) and sample_of(r) in allowed]


# --- quality ---------------------------------------------------------------------------------

def measure(wav_path):
    """Measurable signal properties. Thresholds below are heuristics, not quality guarantees."""
    audio, rate = sf.read(wav_path, dtype='float32', always_2d=True)
    audio = audio.mean(axis=1)
    if not len(audio):
        return {'seconds': 0, 'sample_rate': rate, 'peak': 0, 'clipped': 0, 'silence': 1, 'rms_db': -120}
    frame = max(1, int(rate * .05))
    usable = len(audio) // frame * frame
    frames = audio[:usable].reshape(-1, frame) if usable else audio.reshape(1, -1)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    total_rms = float(np.sqrt(np.mean(audio ** 2)))
    return {'seconds': round(len(audio) / rate, 3), 'sample_rate': rate,
            'peak': round(float(np.max(np.abs(audio))), 4),
            'clipped': round(float(np.mean(np.abs(audio) >= .999)), 5),
            'silence': round(float(np.mean(rms < .005)), 3),
            'rms_db': round(20 * float(np.log10(max(total_rms, 1e-6))), 1)}


def warnings(row):
    quality = row.get('quality') or {}
    found = []
    seconds = quality.get('seconds', row.get('seconds', 0))
    if seconds and seconds < 1.5:
        found.append('very short (< 1.5 s)')
    if seconds > 15:
        found.append('long (> 15 s); may exceed the 1024-token training limit')
    if quality.get('clipped', 0) > .001:
        found.append(f"clipping ({quality['clipped']:.1%} of samples at full scale)")
    if quality.get('silence', 0) > .5:
        found.append(f"mostly silence ({quality['silence']:.0%} of 50 ms frames below -46 dBFS)")
    elif quality.get('rms_db', 0) < -40:
        found.append('very quiet')
    text = (row.get('text') or '').strip()
    if seconds and text:
        rate = len(text) / seconds
        if rate < 4 or rate > 25:
            found.append(f'questionable transcript length ({rate:.0f} characters/s)')
    if not text:
        found.append('empty transcript')
    if row.get('review') == 'needs_review':
        found.append('STT disagreed with subtitle text')
    return found


def analyze(limit=60):
    """Fill in missing/stale clip measurements, a bounded number per call."""
    done = 0
    for path in storage.clips():
        if done >= limit:
            break
        wav = path.with_suffix('.wav')
        if not wav.exists():
            continue
        row = read(path)
        stamp = wav.stat().st_mtime
        if row.get('quality', {}).get('mtime') == stamp:
            continue
        try:
            row['quality'] = dict(measure(wav), mtime=stamp)
        except RuntimeError:
            row['quality'] = {'error': 'unreadable audio', 'mtime': stamp}
        write(path, row)
        done += 1
    return done


def summary():
    registry = load()
    all_clips = [read(p) for p in storage.clips()]
    result = []
    for dataset in registry['datasets']:
        recs = recordings(dataset['id'])
        ids = {r['id'] for r in recs}
        rows = [c for c in all_clips if c and sample_of(c) in ids]
        enabled = [c for c in rows if c.get('enabled', True)]
        reviews = {}
        for c in rows:
            if c.get('source') == 'srt':
                reviews[c.get('review', 'unchecked')] = reviews.get(c.get('review', 'unchecked'), 0) + 1
        rates = sorted({r.get('sample_rate') for r in recs if r.get('sample_rate')})
        result.append(dict(dataset, recordings=len(recs), clips=len(rows), enabled_clips=len(enabled),
                           usable_seconds=round(sum(c.get('seconds', 0) for c in enabled), 1),
                           sample_rates=rates, subtitle_checks=reviews,
                           warnings=sum(1 for c in enabled if warnings(c))))
    return result
