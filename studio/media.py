"""FFmpeg conversion, without shell interpolation or overwriting originals."""
import argparse
import os
import shutil
import subprocess
import uuid
from pathlib import Path


def convert(source, destination, mp3=False):
    executable = shutil.which('ffmpeg')
    if not executable:
        raise RuntimeError('Zainstaluj FFmpeg i dodaj go do PATH (instrukcja w RENTAL.md).')
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError('Plik wynikowy musi być inny niż oryginał.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f'.{uuid.uuid4().hex}{destination.suffix}')
    codec = ['-c:a', 'libmp3lame', '-q:a', '2'] if mp3 else ['-ac', '1', '-ar', '24000', '-c:a', 'pcm_s16le']
    try:
        result = subprocess.run([executable, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                                 '-i', str(source), '-map', '0:a:0', '-vn', *codec, str(temp)],
                                capture_output=True, text=True, timeout=7200,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode:
            raise ValueError('Konwersja nie powiodła się: ' + result.stderr[-1500:])
        # Atomic no-clobber publication, including simultaneous converters.
        os.link(temp, destination)
    finally:
        temp.unlink(missing_ok=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description='Zamień MP4 na MP3. Foldery przetwarzane rekurencyjnie.')
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    files = sorted(source.rglob('*.mp4')) if source.is_dir() else [source]
    failures = 0
    for file in files:
        relative = file.relative_to(source) if source.is_dir() else Path(file.name)
        destination = args.output / relative.with_suffix('.mp3')
        if destination.exists():
            print(f'Pomijam istniejący: {destination}')
            continue
        try:
            print(convert(file, destination, mp3=True))
        except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            failures += 1
            print(f'BŁĄD {file}: {exc}')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
