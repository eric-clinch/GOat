
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import time

def PlayoutListen():
    print("Listening for tensors")
    while True:
        tensor = torch.ones(3)
        dist.recv(tensor=tensor, src=None)
        print(f"Received tensor {tensor}")
        time.sleep(3)
        break

if __name__ == "__main__":
    print("Beginning to init group")
    dist.init_process_group('gloo',
        init_method='tcp://127.0.0.1:23459',
        rank=0,
        world_size=2)
    print("Group init completed")

    listen_proc = Process(target=PlayoutListen)
    listen_proc.start()
    listen_proc.join()
