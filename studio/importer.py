import argparse
import tempfile
from pathlib import Path
from app import import_sample, busy
from .media import convert


def main():
    parser = argparse.ArgumentParser(description='Import local recordings or a folder into the studio.')
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    if busy():
        parser.error('Stop the current studio job first.')
    from .locking import acquire
    compute_lock = acquire()
    paths = sorted(args.source.rglob('*')) if args.source.is_dir() else [args.source]
    accepted = {'.wav', '.flac', '.mp3', '.ogg', '.mp4', '.m4v', '.mov', '.mkv'}
    for path in paths:
        if path.suffix.lower() not in accepted:
            continue
        with tempfile.TemporaryDirectory() as temp:
            wav = convert(path, Path(temp) / 'decoded.wav')
            row = import_sample(wav, path.name)
            print(f"Imported {path.name}: {row['seconds']:.1f} s")


if __name__ == '__main__':
    main()
