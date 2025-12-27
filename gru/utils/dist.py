import os
from typing import Tuple

def ddp_env() -> Tuple[bool, int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return world_size > 1, rank, local_rank, world_size

def init_distributed(backend: str = "nccl"):
    is_ddp, rank, local_rank, world_size = ddp_env()
    if not is_ddp:
        return False, rank, local_rank, world_size
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return True, rank, local_rank, world_size

def barrier():
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0
