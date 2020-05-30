import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import time

if __name__ == "__main__":
    dist.init_process_group('gloo',
        init_method='tcp://127.0.0.1:23459',
        rank=1,
        world_size=2)

    tensor = torch.Tensor([1, 2, 3])
    dist.send(tensor, dst=0)
    time.sleep(3)
