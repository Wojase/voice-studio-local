"""Manual integration check against a RUNNING panel. Continues the newest trainable experiment by
2 updates (bounded), stops, then synthesizes from LATEST through the HTTP API."""
import time
import requests
import soundfile as sf
from studio.storage import DATA

URL = 'http://127.0.0.1:7860/api/'
HEADERS = {'X-Studio': 'local'}


def post(path, body):
    response = requests.post(URL + path, json=body, headers=HEADERS, timeout=30)
    if not response.ok:
        raise RuntimeError(response.json())
    return response.json()


def wait():
    deadline, message = time.monotonic() + 1800, None
    while time.monotonic() < deadline:
        state = requests.get(URL + 'status', timeout=15).json()
        job = state['job']
        if message != job['message']:
            print(job['message'], flush=True)
            message = job['message']
        if not state['busy']:
            assert job['status'] in ('done', 'stopped'), job
            return state
        time.sleep(2)
    raise TimeoutError('Workflow exceeded 30 minutes')


experiments = [e for e in requests.get(URL + 'status', timeout=15).json()['experiments'] if e['resumable'] and not e['archived']]
target = experiments[0]
before = target['updates']
post('jobs', {'action': 'train', 'experiment_id': target['id'], 'run': {'epochs': 1, 'max_updates': 2, 'batch_size': 1,
                                                                       'accumulation': 1, 'lr': 5e-5, 'gpus': target['config']['gpus']}})
state = wait()
after = next(e for e in state['experiments'] if e['id'] == target['id'])['updates']
assert after == before + 2, (before, after)
print(f"PASS: {target['name']} resumed {before} -> {after}", flush=True)
post('jobs', {'action': 'generate', 'texts': ['Hello. This is a test of my local voice model.'],
              'variants': [{'backend': 'kittentts2-lora', 'kind': 'finetuned', 'experiment_id': target['id'], 'checkpoint': 'latest'}]})
result = wait()
audio, rate = sf.read(DATA / 'outputs' / result['job']['output'])
assert len(audio) > rate and abs(audio).max() > .001
print(f'PASS: synthesis from LATEST, {len(audio) / rate:.2f}s at {rate} Hz', flush=True)
