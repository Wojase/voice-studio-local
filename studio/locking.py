"""OS-released exclusive lock across panel and command-line jobs."""
import atexit
import os
from .storage import DATA


def acquire():
    handle = (DATA / '.compute.lock').open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0)
            handle.write(b'0')
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError('Inny proces już pracuje na tych danych. Zatrzymaj go przed uruchomieniem kolejnego.')
    atexit.register(handle.close)
    return handle
