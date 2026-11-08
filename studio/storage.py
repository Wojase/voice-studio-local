import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get('STUDIO_DATA', str(ROOT / 'data'))).resolve()
FOLDERS = ('samples', 'clips', 'features', 'models', 'outputs', 'jobs', 'subtitles', 'experiments', 'exports')
for name in FOLDERS:
    (DATA / name).mkdir(parents=True, exist_ok=True)


def read(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def append_jsonl(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def read_jsonl(path, limit=None):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue  # A torn final line after a crash must not break monitoring.
    return rows[-limit:] if limit else rows


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def clips(root=None):
    return sorted(((root or DATA) / 'clips').glob('*.json'))


def sample_of(row):
    """Original recording a clip was cut from (SRT clips store it; STT clips encode it in the id)."""
    return row.get('sample_id') or row['id'].split('_')[0]
