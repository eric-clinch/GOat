
from torch.multiprocessing import Process, Queue, Value
from resnet.resnet import Resnet
from typing import *

import communication
import torch
import torch.distributed as dist
import time
import argparse
import json
import socket
import pickle
import io
import traceback
import string
import sys

# Given a playout worker socket, listens on that socket for new playouts to
# train on
def ReceivePlayouts(worker: socket.socket, worker_id: int, out_queue: Queue):
    worker.setblocking(True)
    while True:
        try:
            msg: bytes = communication.Receive(worker)
        except Exception as err:
            print(f"Error with worker {worker_id}, ending connection")
            worker.close()
            return

        buffer = io.BytesIO(msg)
        tensor = torch.load(buffer)

        print(f"Received message {tensor}")
        out_queue.put(tensor)


# Listens for new workers to contact the training server. When a new worker
# connects, it a new daemonic process is created to receive playouts from that
# worker. Also checks the network queue and pushes the most
# up-to-date parameters to all of the worker nodes. Uses the shutdown Value
# object to determine when to shut down.
def HandleWorkers(server: socket.socket, playout_queue: Queue,
                  param_queue: Queue, shutdown: Value):
    print("Listening for new workers...")
    server.settimeout(1)  # timeout period of 1 second

    num_workers = 0
    workers: Dict[int, socket.socket] = dict()
    state_dict = None

    while shutdown.value <= 0:
        try:
            worker, _ = server.accept()
            print("Connected to new worker")
            worker_id = num_workers
            worker_proc = Process(target=ReceivePlayouts,
                                  args=(worker, worker_id, out_queue),
                                  daemon=True)
            worker_proc.start()

            if state_dict is not None:
                # Send the new worker the most up-to-date params
                buffer = io.BytesIO()
                torch.save(state_dict, buffer)
                param_bytes = buffer.getvalue()
                print(f"Size of params: {len(param_bytes)} bytes")
                communication.Send(worker, buffer.getvalue())

            workers[worker_id] = worker
            num_workers += 1
        except socket.timeout:
            pass

        if not param_queue.empty():
            # Send the most up-to-date params to all the workers
            state_dict = None
            while not param_queue.empty():
                state_dict = param_queue.get()
            assert(state_dict is not None)

            buffer = io.BytesIO()
            torch.save(state_dict, buffer)
            param_bytes = buffer.getvalue()
            print(f"Size of params: {len(param_bytes)} bytes")
            for worker_id in workers.keys():
                worker: socket.socket = workers[worker_id]
                try:
                    communication.Send(worker, param_bytes)
                except:
                    # Something went wrong with this connection, so remove
                    # this worker
                    print(f"Error with worker {worker_id}, ending connection")
                    workers.pop(worker_id)


# Given a training server socket, listens on that socket for updates to the
# network and places them in the param queue.
def ReceiveParams(server: socket.socket, param_queue: Queue):
    server.setblocking(True)
    print("Listening for network updates...")
    while True:
        try:
            msg: bytes = communication.Receive(server)
        except Exception as err:
            print(f"Error with server connection, ending connection")
            server.close()
            return

        buffer = io.BytesIO(msg)
        state_dict = torch.load(buffer)

        print(f"Received new params: {len(msg)} bytes")
        param_queue.put(state_dict)


def SendPlayout(training_server: socket.socket, tensor: torch.Tensor):
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    communication.Send(training_server, buffer.getvalue())

    print(f"Sent {tensor}, message size {len(buffer.getvalue())}")


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
    port = int(contents['port'])
    rank = int(contents['rank'])

    resnet = Resnet(2, 9)
    net_input = torch.zeros((1, 2, 9, 9))
    net_output = resnet(net_input)
    print(net_output)

    try:
        if rank == 0:
            out_queue = Queue()
            param_queue = Queue()
            shutdown_server = Value('b', 0)

            server: socket.socket = communication.ServerSocket(address, port)

            receiver_proc = Process(target=HandleWorkers,
                                    args=(server, out_queue, param_queue,
                                          shutdown_server))
            receiver_proc.start()

            state_dict = resnet.state_dict()
            print(type(state_dict))
            param_queue.put(state_dict)

            tensor = out_queue.get()
            net_output = resnet(tensor)
            print(net_output)

            print("Shutting down server...")
            with shutdown_server.get_lock():
                shutdown_server.value = 1
        else:
            param_queue = Queue()
            server = communication.WorkerSocket(address, port)
            print("Connected to server")

            receiver_proc = Process(target=ReceiveParams,
                                    args=(server, param_queue),
                                    daemon=True)
            receiver_proc.start()

            # for _ in range(3):
            #     SendPlayout(server)
            #     time.sleep(1)

            state_dict = param_queue.get()
            resnet.load_state_dict(state_dict)
            net_output = resnet(net_input)
            print(net_output)

            tensor = torch.randn(1, 2, 9, 9)
            SendPlayout(server, tensor)

            net_output = resnet(tensor)
            print(net_output)

    except KeyboardInterrupt:
        pass

    print("Shutting down...")
    server.close()
