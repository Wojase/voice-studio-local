"""Start, inspect and gracefully stop the panel without matching unrelated Python jobs."""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from .storage import ROOT, DATA

URL = 'http://127.0.0.1:7860/api/'


def request(path, body=None):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={'Content-Type': 'application/json', 'X-Studio': 'local'})
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.load(response)


def status():
    try:
        return request('status')
    except OSError:
        return None


def stop():
    current = status()
    if current and current['busy']:
        request('stop', {})
        print('Waiting for the running job to stop and save...', flush=True)
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            current = status()
            if not current or not current['busy']:
                break
            time.sleep(2)
        else:
            raise RuntimeError('The job has not stopped. The server was left running to preserve progress.')
    import psutil
    for connection in psutil.net_connections('tcp'):
        if connection.status != 'LISTEN' or connection.laddr.port != 7860:
            continue
        process = psutil.Process(connection.pid)
        if process.cwd() != str(ROOT) or 'app.py' not in process.cmdline():
            raise RuntimeError('Port 7860 belongs to a different program. It was left running.')
        process.terminate()
        process.wait(timeout=20)
        print('Server stopped.')


def start():
    if status():
        print('Server is already running.')
        return
    env = dict(os.environ)
    rental = os.name != 'nt'
    for key, value in {'STUDIO_HOST': '0.0.0.0' if rental else '127.0.0.1', 'STUDIO_PUBLIC': '1' if rental else '0', 'STUDIO_GPUS': '2' if rental else '1',
                       'STUDIO_BATCH_SIZE': '4' if rental else '1', 'STUDIO_STT_MODEL': 'openai/whisper-large-v3' if rental else 'openai/whisper-small.en',
                       'PYTHONUNBUFFERED': '1'}.items():
        env.setdefault(key, value)
    with (DATA / 'panel.log').open('a', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, '-X', 'utf8', 'app.py'], cwd=ROOT, env=env,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                 start_new_session=os.name != 'nt',
                                 creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    for _ in range(60):
        if child.poll() is not None:
            raise RuntimeError('Server exited. Read data/panel.log.')
        if status():
            print('Server ready on port 7860. ' + ('Open your Vast.ai tunnel URL.' if rental else 'Open http://127.0.0.1:7860'))
            return
        time.sleep(1)
    raise RuntimeError('Startup timed out. Read data/panel.log.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'stop', 'restart', 'status'])
    action = parser.parse_args().action
    if action in ('stop', 'restart'):
        stop()
    if action in ('start', 'restart'):
        start()
    if action == 'status':
        current = status()
        summary = {'server': 'offline'} if not current else {
            'server': 'online', 'busy': current['busy'], 'stt': current.get('stt'),
            'experiments': [{k: e.get(k) for k in ('id', 'name', 'status', 'updates', 'best', 'latest')} for e in current.get('experiments', [])],
            'job': {k: current['job'].get(k) for k in ('id', 'action', 'status', 'message')}
            if current.get('job') else None}
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
