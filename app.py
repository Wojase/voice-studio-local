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
    if request.host.split(':')[0] not in ('127.0.0.1', 'localhost') and os.environ.get('STUDIO_PUBLIC', '0') != '1':
        return jsonify(error='Host is not allowed.'), 403
    if request.method != 'GET' and request.headers.get('X-Studio') != 'local':
        return jsonify(error='Open the studio to perform this action.'), 403


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
            job.update(status='error', message='The worker exited unexpectedly. Check its log in data/jobs.')
    return jsonify(job=job, busy=busy(), model=read(DATA / 'models' / 'info.json'),
                   settings={'batch_size': int(os.environ.get('STUDIO_BATCH_SIZE', '1')), 'gpus': int(os.environ.get('STUDIO_GPUS', '1'))},
                   stt={'model': os.environ.get('STUDIO_STT_MODEL', 'openai/whisper-small.en'), 'language': 'en'},
                   trash=[read(p) for p in (DATA / 'trash').glob('*/manifest.json')],
                   subtitles=[read(p) for p in (DATA / 'subtitles').glob('*.json')],
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
        return jsonify(error='GPU status is unavailable.'), 503
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
            raise ValueError('Recordings must be at least 2 seconds long.')
        with sf.SoundFile(dest, 'w', samplerate=incoming.samplerate, channels=1, subtype='PCM_16') as out:
            for block in incoming.blocks(blocksize=65536, dtype='float32', always_2d=True):
                out.write(block.mean(axis=1))
        data = {'id': digest, 'name': original, 'seconds': incoming.frames / incoming.samplerate}
    write(meta, data)
    return data


@app.post('/api/upload')
def upload():
    from studio.locking import acquire
    from studio.media import convert
    from studio.subtitles import parse
    with lock:
        if busy():
            return jsonify(error='Wait for the current job to finish.'), 409
        with acquire():
            file = request.files['file']
            subtitle_file = request.files.get('subtitles')
            mode = request.form.get('subtitle_mode', 'use')
            if subtitle_file and mode not in ('use', 'check'):
                raise ValueError('Choose Use SRT or SRT + check.')
            temp = DATA / 'samples' / (uuid.uuid4().hex + '.upload')
            decoded = temp.with_suffix('.wav')
            new_sample, new_mp3 = None, None
            try:
                file.save(temp)
                video = Path(file.filename).suffix.lower() in ('.mp4', '.m4v', '.mov', '.mkv')
                source = temp
                if video:
                    convert(temp, decoded)
                    source = decoded
                # Validate subtitles before publishing any new dataset files.
                if subtitle_file:
                    content = subtitle_file.stream.read(5 * 1024 * 1024 + 1)
                    if len(content) > 5 * 1024 * 1024:
                        raise ValueError('SRT files must be smaller than 5 MB.')
                    parse(content.decode('utf-8-sig'), sf.info(source).duration)
                    subtitle_file.stream.seek(0)
                if video:
                    converted = DATA / 'converted'
                    converted.mkdir(exist_ok=True)
                    with temp.open('rb') as stream:
                        key = hashlib.file_digest(stream, 'sha256').hexdigest()[:24]
                    mp3 = converted / (key + '.mp3')
                    if not mp3.exists():
                        convert(temp, mp3, mp3=True)
                        new_mp3 = mp3
                with source.open('rb') as stream:
                    sample_id = hashlib.file_digest(stream, 'sha256').hexdigest()[:24]
                if not (DATA / 'samples' / (sample_id + '.json')).exists():
                    new_sample = sample_id
                result = import_sample(source, Path(file.filename).name)
                if subtitle_file:
                    result['subtitles'] = install_subtitles(result['id'], subtitle_file, trusted=mode == 'use')
                    result['check_requested'] = mode == 'check'
                return jsonify(result)
            except Exception:
                if new_sample:
                    for suffix in ('.json', '.wav'):
                        (DATA / 'samples' / (new_sample + suffix)).unlink(missing_ok=True)
                if new_mp3:
                    new_mp3.unlink(missing_ok=True)
                raise
            finally:
                temp.unlink(missing_ok=True)
                decoded.unlink(missing_ok=True)


@app.post('/api/clips/<key>')
def edit_clip(key):
    with lock:
        if busy():
            return jsonify(error='Stop the current job before editing clips.'), 409
        path = DATA / 'clips' / f'{key}.json'
        if not path.exists():
            return jsonify(error='Clip not found.'), 404
        payload = request.get_json()
        text = str(payload.get('text', '')).strip()
        if not text or len(text) > 2000:
            raise ValueError('Clip text must contain 1-2000 characters.')
        row = read(path)
        row.update(text=text, enabled=bool(payload.get('enabled', True)))
        if row.get('source') == 'srt':
            row['review'] = 'manual'
        write(path, row)
        return jsonify(row)


@app.post('/api/jobs')
def start():
    global active, process
    with lock:
        if busy():
            return jsonify(error='A job is already using the GPU.'), 409
        payload = request.get_json()
        action = payload.get('action')
        if action not in ('prepare', 'train', 'generate', 'check_subtitles'):
            raise ValueError('Unknown job.')
        args = {}
        if action == 'check_subtitles':
            from studio.management import safe_key
            ids = payload.get('sample_ids', [payload.get('sample_id')])
            if not isinstance(ids, list) or not 1 <= len(ids) <= 500:
                raise ValueError('Choose 1-500 recordings to check.')
            args['sample_ids'] = list(dict.fromkeys(safe_key(key) for key in ids))
            for key in args['sample_ids']:
                if not (DATA / 'subtitles' / (key + '.json')).exists():
                    raise ValueError('Attach an SRT file to this recording first.')
        if action == 'train':
            args = {'epochs': int(payload.get('epochs', 3)), 'accumulation': int(payload.get('accumulation', 4)),
                    'lr': float(payload.get('lr', .0001)), 'batch_size': int(payload.get('batch_size', os.environ.get('STUDIO_BATCH_SIZE', '1')))}
            if not 1 <= args['epochs'] <= 100 or not 1 <= args['accumulation'] <= 32 or not 1e-6 <= args['lr'] <= 1e-3:
                raise ValueError('Invalid training settings.')
            if not 1 <= args['batch_size'] <= 32:
                raise ValueError('Batch size must be between 1 and 32 per GPU.')
        if action == 'generate':
            args['text'] = str(payload.get('text', '')).strip()
            if not 1 <= len(args['text']) <= 3000:
                raise ValueError('Enter 1-3000 characters.')
        active = uuid.uuid4().hex
        write(DATA / 'jobs' / f'{active}.json', {'id': active, 'action': action, 'args': args,
                                               'status': 'running', 'message': 'Starting...', 'progress': 0})
        with open(DATA / 'jobs' / f'{active}.log', 'w', encoding='utf-8') as log:
            count = int(os.environ.get('STUDIO_GPUS', '1')) if action == 'train' else 1
            command = [sys.executable, '-X', 'utf8', '-m', 'studio.worker', active]
            if count > 1:
                if sys.platform == 'win32':
                    raise ValueError('Multi-GPU training requires Linux with NCCL.')
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
    return jsonify(message='Stopping after the current update. Saving may take a moment.')


@app.get('/audio/<kind>/<name>')
def audio(kind, name):
    if kind not in ('clips', 'outputs', 'samples', 'converted') or not name.endswith(('.wav', '.mp3')):
        return '', 404
    return send_from_directory(DATA / kind, name)


@app.post('/api/manage')
def manage():
    from studio import management
    from studio.locking import acquire
    with lock:
        if busy():
            return jsonify(error='Stop the current job before changing data.'), 409
        body = request.get_json()
        with acquire():
            if body.get('action') == 'purge':
                if body.get('confirm') != 'DELETE':
                    raise ValueError('Permanent deletion requires typing DELETE.')
                management.purge(body.get('key', ''))
                return jsonify(deleted=True)
            if body.get('action') == 'restore':
                return jsonify(management.restore(body.get('key', '')))
            paths = management.collect(body.get('kind'), body.get('key', ''))
            return jsonify(management.trash(paths, body.get('kind') + ': ' + body.get('key', 'all')))


def install_subtitles(sample_id, subtitle_file, trusted=False):
    import tempfile
    from studio.subtitles import parse, grouped
    from studio import management
    management.safe_key(sample_id)
    source = DATA / 'samples' / (sample_id + '.wav')
    if not source.is_file():
        raise ValueError('Recording not found.')
    content = subtitle_file.stream.read(5 * 1024 * 1024 + 1)
    if len(content) > 5 * 1024 * 1024:
        raise ValueError('SRT files must be smaller than 5 MB.')
    text = content.decode('utf-8-sig')
    rows = parse(text, sf.info(source).duration)
    segments = grouped(rows)
    fingerprint = hashlib.sha256(content).hexdigest()[:10]
    with tempfile.TemporaryDirectory(dir=DATA) as temp:
        staged = Path(temp)
        with sf.SoundFile(source) as incoming:
            for i, row in enumerate(segments):
                key = f'{sample_id}_srt{fingerprint}_{i:05d}'
                incoming.seek(round(row['start'] * incoming.samplerate))
                audio_data = incoming.read(round((row['end'] - row['start']) * incoming.samplerate))
                sf.write(staged / (key + '.wav'), audio_data, incoming.samplerate, subtype='PCM_16')
                write(staged / (key + '.json'), dict(row, id=key, sample_id=sample_id,
                      seconds=len(audio_data) / incoming.samplerate, enabled=trusted,
                      source='srt', review='provided' if trusted else 'unchecked', check=None))
        old = list((DATA / 'clips').glob(sample_id + '_*')) + list((DATA / 'features').glob(sample_id + '_*'))
        old += list((DATA / 'subtitles').glob(sample_id + '.*'))
        backup = management.trash(old, 'Replace subtitles: ' + sample_id) if old else None
        installed = []
        try:
            for file in staged.iterdir():
                target = DATA / 'clips' / file.name
                file.rename(target)
                installed.append(target)
            (DATA / 'subtitles').mkdir(exist_ok=True)
            result = {'sample_id': sample_id, 'filename': Path(subtitle_file.filename).name,
                      'cues': len(rows), 'clips': len(segments), 'fingerprint': fingerprint}
            write(DATA / 'subtitles' / (sample_id + '.json'), result)
        except Exception:
            for file in installed:
                file.unlink(missing_ok=True)
            (DATA / 'subtitles' / (sample_id + '.json')).unlink(missing_ok=True)
            if backup:
                management.restore(backup['id'])
            raise
    return result


@app.post('/api/subtitles/<sample_id>')
def attach_subtitles(sample_id):
    from studio.locking import acquire
    with lock:
        if busy():
            return jsonify(error='Stop the current job before importing subtitles.'), 409
        with acquire():
            return jsonify(install_subtitles(sample_id, request.files['file']))


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
    serve(app, host=os.environ.get('STUDIO_HOST', '127.0.0.1'), port=7860, threads=6, max_request_body_size=2 * 1024 ** 3)
