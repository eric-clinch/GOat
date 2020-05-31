
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import time
import argparse
import json


def PlayoutListen():
    print("Listening for tensors")
    while True:
        tensor = torch.ones(3)
        dist.recv(tensor=tensor, src=None)
        print(f"Received tensor {tensor}")
        time.sleep(3)
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hosts a distributed pytorch system")
    parser.add_argument('config_file', type=str, nargs="?",
                        help="Location of the json config file")
    args = parser.parse_args()

    with open(args.config_file) as config_file:
        contents = json.load(config_file)

    if ('addr' not in contents) or ('port' not in contents) or ('rank' not in contents):
        print("IP address (addr) and port number required in config")

    address = contents['addr']
    port = contents['port']
    rank = int(contents['rank'])

    print("Beginning to init group")
    connection = f"tcp://{address}:{port}"
    if rank == 0:
        print(f"Hosting at {connection}")
    else:
        print(f"Connecting to {connection}")
    dist.init_process_group('gloo',
                            init_method=connection,
                            rank=rank,
                            world_size=2)
    print("Group init completed")

    if rank == 0:
        listen_proc = Process(target=PlayoutListen)
        listen_proc.start()
        listen_proc.join()
    else:
        tensor = torch.Tensor([1, 2, 3])
        dist.send(tensor, dst=0)
        print("Tensor sent to host")
