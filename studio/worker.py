import sys
import os
import traceback
from .storage import DATA, read, write


def main():
    job_id = sys.argv[1]
    path = DATA / 'jobs' / f'{job_id}.json'
    job = read(path)
    from . import distributed as parallel
    parallel.initialize()
    from .locking import acquire
    if parallel.rank() == 0:
        compute_lock = acquire()
    job['pid'] = os.getpid()
    if parallel.rank() == 0:
        write(path, job)
    def report(message, **fields):
        if parallel.rank() != 0:
            return
        job.update(message=message, **fields)
        write(path, job)
        print(message, flush=True)
    def stopped():
        return (DATA / 'jobs' / f'{job_id}.stop').exists()
    try:
        from . import engine
        getattr(engine, job['action'])(report, stopped, **job['args'])
        job['status'] = 'stopped' if stopped() else 'done'
        if stopped():
            job['message'] = 'Zatrzymano. Zapisany postęp pozostaje dostępny.'
    except Exception as exc:
        traceback.print_exc()
        message = str(exc)
        if 'out of memory' in message.lower():
            message = 'Brak wolnej pamięci GPU. Zamknij inne programy używające karty i spróbuj ponownie. Ostatni ukończony zapis modelu pozostał na dysku.'
        job.update(status='error', message=message)
        if parallel.world() > 1:
            if parallel.rank() == 0:
                write(path, job)
            raise
    if parallel.rank() == 0:
        write(path, job)


if __name__ == '__main__':
    main()
