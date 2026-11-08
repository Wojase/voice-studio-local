import hashlib
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import soundfile as sf
import psutil
from flask import Flask, jsonify, request, send_from_directory
from studio.storage import ROOT, DATA, read, write, clips

app = Flask(__name__, static_folder='web')
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 ** 3
lock = threading.Lock()
previous_jobs = sorted((DATA / 'jobs').glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
active = previous_jobs[0].stem if previous_jobs else None
process = None


def busy():
    if process is not None:
        return process.poll() is None
    job = read(DATA / 'jobs' / f'{active}.json', {}) if active else {}
    if job.get('status') != 'running' or not job.get('pid'):
        return False
    try:
        command = psutil.Process(job['pid']).cmdline()
        return 'studio.worker' in command and active in command
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


@app.before_request
def local_only():
    if request.host.split(':')[0] not in ('127.0.0.1', 'localhost'):
        return jsonify(error='Niedozwolony adres.'), 403
    if request.method != 'GET' and request.headers.get('X-Studio') != 'local':
        return jsonify(error='Otwórz panel lokalny, aby wykonać tę operację.'), 403


@app.errorhandler(Exception)
def error(exc):
    from werkzeug.exceptions import HTTPException
    return jsonify(error=str(exc)), exc.code if isinstance(exc, HTTPException) else 400


@app.get('/')
def home():
    return send_from_directory(ROOT / 'web', 'index.html')


@app.get('/api/status')
def status():
    job = read(DATA / 'jobs' / f'{active}.json') if active else None
    if job and job['status'] == 'running' and not busy():
        job = read(DATA / 'jobs' / f'{active}.json')
        if job['status'] == 'running':
            job.update(status='error', message='Proces zakończył się nieoczekiwanie. Sprawdź dziennik w data/jobs.')
    return jsonify(job=job, busy=busy(), model=read(DATA / 'models' / 'info.json'),
                   settings={'batch_size': int(os.environ.get('STUDIO_BATCH_SIZE', '1')), 'gpus': int(os.environ.get('STUDIO_GPUS', '1'))},
                   converted=[p.name for p in sorted((DATA / 'converted').glob('*.mp3'))],
                   samples=[read(p) for p in sorted((DATA / 'samples').glob('*.json'))],
                   clips=[read(p) for p in clips()],
                   outputs=[p.name for p in sorted((DATA / 'outputs').glob('*.wav'), key=lambda p: p.stat().st_mtime, reverse=True)][:20])


@app.get('/api/gpu')
def gpu():
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=5)
    if result.returncode:
        return jsonify(error='Nie można odczytać GPU.'), 503
    cards = []
    for line in result.stdout.splitlines():
        values = [v.strip() for v in line.split(',')]
        cards.append(dict(name=values[0], use=int(values[1]), used=int(values[2]), total=int(values[3]), temperature=int(values[4])))
    return jsonify(**cards[0], cards=cards)


def import_sample(source, original):
    with open(source, 'rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()[:24]
    dest = DATA / 'samples' / f'{digest}.wav'
    meta = dest.with_suffix('.json')
    if meta.exists():
        return read(meta)
    # Decode before accepting: the extension alone does not establish an audio file.
    with sf.SoundFile(source) as incoming:
        if incoming.frames < incoming.samplerate * 2:
            raise ValueError('Nagranie musi mieć przynajmniej 2 sekundy.')
        with sf.SoundFile(dest, 'w', samplerate=incoming.samplerate, channels=1, subtype='PCM_16') as out:
            for block in incoming.blocks(blocksize=65536, dtype='float32', always_2d=True):
                out.write(block.mean(axis=1))
        data = {'id': digest, 'name': original, 'seconds': incoming.frames / incoming.samplerate}
    write(meta, data)
    return data


@app.post('/api/upload')
def upload():
    with lock:
        if busy():
            return jsonify(error='Poczekaj na zakończenie bieżącego zadania.'), 409
        file = request.files['file']
        temp = DATA / 'samples' / (uuid.uuid4().hex + '.upload')
        try:
            file.save(temp)
            if Path(file.filename).suffix.lower() in ('.mp4', '.m4v', '.mov', '.mkv'):
                from studio.media import convert
                converted = DATA / 'converted'
                converted.mkdir(exist_ok=True)
                with temp.open('rb') as stream:
                    key = hashlib.file_digest(stream, 'sha256').hexdigest()[:24]
                mp3 = converted / (key + '.mp3')
                if not mp3.exists():
                    convert(temp, mp3, mp3=True)
                decoded = temp.with_suffix('.wav')
                try:
                    convert(temp, decoded)
                    return jsonify(import_sample(decoded, Path(file.filename).name))
                finally:
                    decoded.unlink(missing_ok=True)
            return jsonify(import_sample(temp, Path(file.filename).name))
        finally:
            temp.unlink(missing_ok=True)


@app.post('/api/clips/<key>')
def edit_clip(key):
    with lock:
        if busy():
            return jsonify(error='Nie można zmieniać próbek podczas pracy.'), 409
        path = DATA / 'clips' / f'{key}.json'
        if not path.exists():
            return jsonify(error='Nie ma takiego fragmentu.'), 404
        payload = request.get_json()
        text = str(payload.get('text', '')).strip()
        if not text or len(text) > 2000:
            raise ValueError('Tekst fragmentu musi mieć 1–2000 znaków.')
        row = read(path)
        row.update(text=text, enabled=bool(payload.get('enabled', True)))
        write(path, row)
        return jsonify(row)


@app.post('/api/jobs')
def start():
    global active, process
    with lock:
        if busy():
            return jsonify(error='Jedno zadanie już korzysta z GPU.'), 409
        payload = request.get_json()
        action = payload.get('action')
        if action not in ('prepare', 'train', 'generate'):
            raise ValueError('Nieznane zadanie.')
        args = {}
        if action == 'train':
            args = {'epochs': int(payload.get('epochs', 3)), 'accumulation': int(payload.get('accumulation', 4)),
                    'lr': float(payload.get('lr', .0001)), 'batch_size': int(payload.get('batch_size', os.environ.get('STUDIO_BATCH_SIZE', '1')))}
            if not 1 <= args['epochs'] <= 100 or not 1 <= args['accumulation'] <= 32 or not 1e-6 <= args['lr'] <= 1e-3:
                raise ValueError('Niepoprawne ustawienia treningu.')
            if not 1 <= args['batch_size'] <= 32:
                raise ValueError('Batch musi mieć od 1 do 32 próbek na GPU.')
        if action == 'generate':
            args['text'] = str(payload.get('text', '')).strip()
            if not 1 <= len(args['text']) <= 3000:
                raise ValueError('Podaj od 1 do 3000 znaków.')
        active = uuid.uuid4().hex
        write(DATA / 'jobs' / f'{active}.json', {'id': active, 'action': action, 'args': args,
                                               'status': 'running', 'message': 'Uruchamianie…', 'progress': 0})
        with open(DATA / 'jobs' / f'{active}.log', 'w', encoding='utf-8') as log:
            count = int(os.environ.get('STUDIO_GPUS', '1')) if action == 'train' else 1
            command = [sys.executable, '-X', 'utf8', '-m', 'studio.worker', active]
            if count > 1:
                if sys.platform == 'win32':
                    raise ValueError('Trening na wielu GPU wymaga Linuksa z NCCL.')
                command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                           f'--nproc_per_node={count}', '--module', 'studio.worker', active]
            process = subprocess.Popen(command, cwd=ROOT,
                                       stdout=log, stderr=log, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return jsonify(id=active)


@app.post('/api/stop')
def stop():
    with lock:
        if busy():
            (DATA / 'jobs' / f'{active}.stop').touch()
    return jsonify(message='Zatrzymanie po bieżącym kroku; zapis treningu może chwilę potrwać.')


@app.get('/audio/<kind>/<name>')
def audio(kind, name):
    if kind not in ('clips', 'outputs', 'samples', 'converted') or not name.endswith(('.wav', '.mp3')):
        return '', 404
    return send_from_directory(DATA / kind, name)


if __name__ == '__main__':
    from waitress import serve
    if '--open' in sys.argv:
        import urllib.request
        import webbrowser
        import time
        address = 'http://127.0.0.1:7860'
        try:
            with urllib.request.urlopen(address + '/api/status', timeout=2) as response:
                if response.status == 200:
                    webbrowser.open(address)
                    sys.exit(0)
        except OSError:
            pass
        def open_when_ready():
            for _ in range(30):
                try:
                    with urllib.request.urlopen(address, timeout=1):
                        webbrowser.open(address)
                        return
                except OSError:
                    time.sleep(.2)
        threading.Thread(target=open_when_ready, daemon=True).start()
    print('Panel: http://127.0.0.1:7860', flush=True)
    serve(app, host='127.0.0.1', port=7860, threads=6, max_request_body_size=2 * 1024 ** 3)
