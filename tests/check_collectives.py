"""Two CPU processes test the exact gradient-reduction function used on GPU."""
import sys
import torch
import torch.distributed as dist
from studio.distributed import synchronize

rank = int(sys.argv[1])
dist.init_process_group('gloo', init_method=sys.argv[2], rank=rank, world_size=2)
p = torch.nn.Parameter(torch.ones(3))
p.grad = torch.full((3,), float(rank + 1))
synchronize([p])
assert torch.allclose(p.grad, torch.full((3,), 1.5))
dist.destroy_process_group()
print(f'PASS rank {rank}: synchronized gradients')
