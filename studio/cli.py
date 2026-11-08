"""Headless commands, also launched through torchrun on a rented server.

python -m studio.cli experiments                       # list experiments and checkpoints
python -m studio.cli create --name "My voice"          # new experiment (fixed settings: --rank --alpha --targets ...)
python -m studio.cli train --experiment ID --epochs 1 --max-updates 10
python -m studio.cli generate --experiment ID --checkpoint best --text "Hello."
python -m studio.cli export --experiment ID --checkpoint best
"""
import argparse
import json
import signal

from . import distributed as parallel


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='action', required=True)
    commands.add_parser('prepare')
    commands.add_parser('experiments')
    create = commands.add_parser('create')
    create.add_argument('--name', required=True)
    create.add_argument('--dataset', default='default')
    create.add_argument('--speaker', default=None)
    create.add_argument('--rank', type=int, default=8)
    create.add_argument('--alpha', type=float, default=16.0)
    create.add_argument('--dropout', type=float, default=0.0)
    create.add_argument('--targets', default='q_proj,v_proj')
    create.add_argument('--seed', type=int, default=42)
    create.add_argument('--validation-fraction', type=float, default=.1)
    train = commands.add_parser('train')
    train.add_argument('--experiment', default=None, help='Experiment ID (default: newest).')
    train.add_argument('--epochs', type=int, default=3)
    train.add_argument('--max-updates', type=int, default=0)
    train.add_argument('--batch-size', type=int, default=4)
    train.add_argument('--accumulation', type=int, default=2)
    train.add_argument('--lr', type=float, default=.00005)
    train.add_argument('--warmup-updates', type=int, default=0)
    train.add_argument('--scheduler', default='constant')
    train.add_argument('--validate-every', type=int, default=0)
    train.add_argument('--save-every', type=int, default=25)
    generate = commands.add_parser('generate')
    generate.add_argument('--experiment', default=None)
    generate.add_argument('--checkpoint', default='best', help='best, latest or a snapshot file name')
    generate.add_argument('--text', required=True)
    generate.add_argument('--seed', type=int, default=None)
    export = commands.add_parser('export')
    export.add_argument('--experiment', required=True)
    export.add_argument('--checkpoint', default='best')
    export.add_argument('--no-reference', action='store_true')
    args = parser.parse_args()
    from . import registry
    registry.migrate()

    if args.action == 'experiments':
        for e in registry.listing():
            print(f"{e['id']}  {e['name']}  [{e['status']}]  updates={e['updates']}  best={e.get('best')}  "
                  f"checkpoints={', '.join(e.get('checkpoints', {}))}")
        return
    if args.action == 'create':
        from . import datasets
        speaker = args.speaker or datasets.find(args.dataset)['speakers'][0]
        e = registry.create(args.name, args.dataset, speaker, {
            'rank': args.rank, 'alpha': args.alpha, 'dropout': args.dropout, 'targets': args.targets.split(','),
            'seed': args.seed, 'validation_fraction': args.validation_fraction})
        print(e['id'])
        return
    if args.action == 'export':
        from .export import export as build
        print(build(args.experiment, args.checkpoint, include_reference=not args.no_reference))
        return

    experiment = args.experiment if hasattr(args, 'experiment') and args.experiment else None
    if args.action in ('train', 'generate') and not experiment:
        rows = [e for e in registry.listing(include_archived=False) if args.action == 'train' or e.get('latest')]
        if not rows:
            parser.error('No experiment found. Create one first.')
        experiment = rows[0]['id']
    parallel.initialize()
    if parallel.rank() == 0:
        from .locking import acquire
        compute_lock = acquire()  # noqa: F841  Held for the lifetime of the process.
    if parallel.world() > 1 and args.action != 'train':
        parser.error('Multiple GPUs are supported for training. Run preparation and generation as a single process.')
    from . import engine
    stop = False

    def interrupt(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)

    def report(message, **fields):
        if parallel.rank() == 0:
            print(message, json.dumps({k: v for k, v in fields.items() if k != 'experiment_id'}), flush=True)

    if args.action == 'prepare':
        engine.prepare(report, lambda: stop)
    elif args.action == 'train':
        run = {'epochs': args.epochs, 'max_updates': args.max_updates, 'batch_size': args.batch_size,
               'accumulation': args.accumulation, 'lr': args.lr, 'warmup_updates': args.warmup_updates,
               'scheduler': args.scheduler, 'validate_every': args.validate_every, 'save_every': args.save_every,
               'gpus': list(range(parallel.world()))}
        engine.train(report, lambda: stop, experiment_id=experiment, run=run)
    else:
        variant = {'backend': 'kittentts2-lora', 'kind': 'finetuned', 'experiment_id': experiment,
                   'checkpoint': args.checkpoint, 'seed': args.seed}
        from .generation import validate_variant
        engine.generate(report, lambda: stop, variants=[validate_variant(variant)], texts=[args.text])


if __name__ == '__main__':
    main()
