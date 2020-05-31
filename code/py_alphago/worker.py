import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import time
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Works in a distributed pytorch system")
    parser.add_argument('config_file', type=str, nargs="?",
                        help="Location of the json config file")
    args = parser.parse_args()

    with open(args.config_file) as config_file:
        contents = json.load(config_file)

    if ('addr' not in contents) or ('port' not in contents):
        print("IP address (addr) and port number required in config")

    address = contents['addr']
    port = contents['port']

    connection = f"tcp://{address}:{port}"
    print(f"Connecting to {connection}")
    dist.init_process_group('gloo',
                            init_method=connection,
                            rank=1,
                            world_size=2)

    tensor = torch.Tensor([1, 2, 3])
    dist.send(tensor, dst=0)
    time.sleep(3)
