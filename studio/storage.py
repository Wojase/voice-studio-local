import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get('STUDIO_DATA', str(ROOT / 'data'))).resolve()
for name in ('samples', 'clips', 'features', 'models', 'outputs', 'jobs'):
    (DATA / name).mkdir(parents=True, exist_ok=True)


def read(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def clips():
    return sorted((DATA / 'clips').glob('*.json'))
