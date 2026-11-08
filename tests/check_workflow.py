"""Manual integration check. Continues the user's model by one update, then synthesizes."""
import time
import requests
import soundfile as sf
from studio.storage import DATA

URL = 'http://127.0.0.1:7860/api/'
HEADERS = {'X-Studio': 'local'}


def post(path, body):
    response = requests.post(URL + path, json=body, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return response.json()


def wait(stop_after=None):
    deadline = time.monotonic() + 1800
    message = None
    stop_sent = False
    while time.monotonic() < deadline:
        state = requests.get(URL + 'status', timeout=15).json()
        job = state['job']
        if message != job['message']:
            print(job['message'], flush=True)
            message = job['message']
        if stop_after is not None and job.get('step', 0) >= stop_after and not stop_sent:
            post('stop', {})
            stop_sent = True
        if not state['busy']:
            assert job['status'] in ('done', 'stopped'), job
            return state
        time.sleep(2)
    raise TimeoutError('Workflow exceeded 30 minutes')


before = requests.get(URL + 'status', timeout=15).json()['model']['steps']
post('jobs', {'action': 'train', 'epochs': 1, 'accumulation': 1, 'lr': .00005})
after = wait(stop_after=before + 1)
assert after['model']['steps'] > before
print(f"PASS: resume {before} -> {after['model']['steps']}, stop saved", flush=True)
post('jobs', {'action': 'generate', 'text': 'Dzień dobry. To jest próba mojego lokalnego modelu głosu.'})
result = wait()
assert result['model']['steps'] == after['model']['steps'], 'Generation must not train'
path = DATA / 'outputs' / result['job']['output']
audio, rate = sf.read(path)
assert len(audio) > rate
assert abs(audio).max() > .001
print(f'PASS: saved-model synthesis, {len(audio)/rate:.2f}s at {rate}Hz -> {path}', flush=True)
