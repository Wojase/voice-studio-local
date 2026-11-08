"""Headless commands, also launched through torchrun on a rented server."""
import argparse
import signal
from . import distributed as parallel


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='action', required=True)
    commands.add_parser('prepare')
    train = commands.add_parser('train')
    train.add_argument('--epochs', type=int, default=3)
    train.add_argument('--batch-size', type=int, default=4)
    train.add_argument('--accumulation', type=int, default=2)
    train.add_argument('--lr', type=float, default=.00005)
    generate = commands.add_parser('generate')
    generate.add_argument('--text', required=True)
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'train' and (not 1 <= args['batch_size'] <= 32 or not 1 <= args['epochs'] <= 100
                              or not 1 <= args['accumulation'] <= 32 or not 1e-6 <= args['lr'] <= .001):
        parser.error('Invalid training settings.')
    parallel.initialize()
    if parallel.rank() == 0:
        from .locking import acquire
        compute_lock = acquire()
    from . import engine
    stop = False
    def interrupt(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    def report(message, **fields):
        if parallel.rank() == 0:
            print(message, fields, flush=True)
    if parallel.world() > 1 and action != 'train':
        parser.error('Multiple GPUs are supported for training. Run preparation and generation as a single process.')
    getattr(engine, action)(report, lambda: stop, **args)


if __name__ == '__main__':
    main()
