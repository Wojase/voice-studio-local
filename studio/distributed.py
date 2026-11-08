"""One training replica per GPU, synchronized adapter gradients only."""
import os
import torch
import torch.distributed as dist


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world():
    return dist.get_world_size() if dist.is_initialized() else 1


def initialize():
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('nccl')


def barrier():
    if world() > 1:
        dist.barrier()


def should_stop(stopped):
    value = torch.tensor(int(stopped()) if rank() == 0 else 0, device='cuda')
    if world() > 1:
        dist.broadcast(value, src=0)
    return bool(value.item())


def synchronize(parameters):
    # Fuse small LoRA gradients into a single collective per update.
    parameters = list(parameters)
    flat = torch.cat([p.grad.reshape(-1) for p in parameters])
    if world() > 1:
        dist.all_reduce(flat)
        flat.div_(world())
    offset = 0
    for p in parameters:
        p.grad.copy_(flat[offset:offset + p.numel()].view_as(p))
        offset += p.numel()


def groups(indices, accumulation, batch_size, rank_id, world_size):
    """Identical update counts; last group padded like DistributedSampler."""
    size = accumulation * batch_size * world_size
    for offset in range(0, len(indices), size):
        group = indices[offset:offset + size]
        if world_size > 1:
            group += [indices[i % len(indices)] for i in range(size - len(group))]
        local = group[rank_id::world_size]
        yield offset, [local[i:i + batch_size] for i in range(0, len(local), batch_size)]
