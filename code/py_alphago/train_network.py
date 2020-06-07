from resnet.resnet import Resnet, DEVICE
from torch.utils.data import DataLoader
from torch.multiprocessing import Process, Queue, Lock, Value
from play_games import MoveDatapoint, DELIM, Send, StripDelim
from typing import *

import torch
import torch.optim as Optimizer
import pickle
import time
import math
import random
import argparse
import io
import socket
import json


# taken from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, point):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = point
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# -----------------------------------------------------------
# Socket communication code
# -----------------------------------------------------------

# Given a playout worker socket, listens on that socket for new playouts to
# train on
def ReceivePlayouts(worker: socket.socket, worker_id: int,
                    replay_memory: ReplayMemory, mem_lock: Lock):
    worker.setblocking(True)
    msg = b''
    while True:
        try:
            msg += worker.recv(8192)
        except Exception as err:
            print(f"Error with worker {worker_id}, ending connection")
            worker.close()
            return

        if msg.endswith(DELIM):
            msg = StripDelim(msg)
            playout: List[MoveDatapoint] = pickle.loads(msg)

            mem_lock.acquire()
            for move_datapoint in playout:
                replay_memory.push(move_datapoint)
            mem_lock.release()
            print(f"{len(playout)} new datapoints added from worker {worker_id}")

            msg = b''


# Listens for new workers to contact the training server. When a new worker
# connects, it a new daemonic process is created to receive playouts from that
# worker. Also checks the network queue and pushes the most
# up-to-date parameters to all of the worker nodes. Uses the shutdown Value
# object to determine when to shut down.
def HandleWorkers(server: socket.socket, replay_memory: ReplayMemory,
                  mem_lock: Lock, param_queue: Queue, shutdown: Value):
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
                                  args=(worker, worker_id,
                                        replay_memory, mem_lock),
                                  daemon=True)
            worker_proc.start()

            if state_dict is not None:
                # Send the new worker the most up-to-date params
                buffer = io.BytesIO()
                torch.save(state_dict, buffer)
                param_bytes = buffer.getvalue()
                Send(worker, buffer.getvalue())

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
            print("Sending new params to workers")
            for worker_id in workers.keys():
                worker: socket.socket = workers[worker_id]
                try:
                    Send(worker, param_bytes)
                except:
                    # Something went wrong with this connection, so remove
                    # this worker
                    print(f"Error with worker {worker_id}, ending connection")
                    workers.pop(worker_id)

# -----------------------------------------------------------
# Training code
# -----------------------------------------------------------


# Returns an object that is pickled in the given file
def LoadObject(file_name: str):
    file = open(file_name, 'rb')
    obj = pickle.load(file)
    file.close()
    return obj


def GetPlayerStoneMap(board_list: List[List[str]], player: int) -> List[List[int]]:
    player_str = 'B' if player == 0 else 'W'
    def stone_mapper(x): return 1 if x == player_str else 0
    stone_map = [list(map(stone_mapper, row)) for row in board_list]
    return stone_map


def GetDataset(replay_memory: ReplayMemory, mem_lock: Lock,
               batchsize: int) -> DataLoader:
    datapoints = []
    mem_lock.acquire()
    for move_datapoint in replay_memory.memory:
        player = move_datapoint.player
        player_stone_map = GetPlayerStoneMap(
            move_datapoint.board_list, player)
        opponent_stone_map = GetPlayerStoneMap(
            move_datapoint.board_list, 1 - player)
        network_input = torch.Tensor(
            [player_stone_map, opponent_stone_map])
        value = torch.Tensor([move_datapoint.confidence])
        policy = torch.Tensor(move_datapoint.policy)
        datapoints.append((network_input, value, policy))
    mem_lock.release()
    return DataLoader(datapoints, batch_size=batchsize, shuffle=True)


def AugmentData(states, policies):
    board_moves = policies[:, :-1]
    pass_probs = policies[:, -1].unsqueeze(1)
    shape = board_moves.shape
    batchsize = shape[0]
    board_size = int(math.sqrt(shape[1]))
    board_moves = board_moves.reshape([batchsize, board_size, board_size])

    if random.randint(0, 1) == 1:
        # flip the tensors
        states = states.flip([-2, -1])
        board_moves = board_moves.flip([-2, -1])
    rotations = random.randint(0, 3)
    for _ in range(rotations):
        states = states.rot90(1, [-2, -1])
        board_moves = board_moves.rot90(1, [-2, -1])

    board_moves = board_moves.flatten(start_dim=1)
    policies = torch.cat([board_moves, pass_probs], 1)

    return states, policies


def CrossEntropy(target: torch.Tensor, distribution: torch.Tensor) -> torch.Tensor:
    losses = torch.einsum('ij,ij->i', target, torch.log(distribution))
    return -torch.mean(losses)


def Epoch(nn: Resnet, data_loader: DataLoader, value_criterion,
          policy_criterion, optim: Optimizer) -> float:
    running_loss = 0
    count = 0
    for batch in data_loader:
        optim.zero_grad()
        states, target_values, target_policies = batch
        states, target_policies = AugmentData(states, target_policies)

        states = states.to(DEVICE)
        target_values = target_values.to(DEVICE)
        target_policies = target_policies.to(DEVICE)
        nn_values, nn_policies = nn(states)
        value_loss = value_criterion(nn_values, target_values)
        policy_loss = policy_criterion(target_policies, nn_policies)
        loss = value_loss + policy_loss
        loss.backward()
        optim.step()

        running_loss += loss.item()
        count += 1

    return running_loss / count


def Train(net: Resnet, replay_memory: ReplayMemory, mem_lock: Lock,
          save_path: str, epochs: int = 500):
    print(f"Training on {len(replay_memory)} datapoints")
    batch_size = 1024
    data_loader = GetDataset(replay_memory, mem_lock, batch_size)

    value_criterion = torch.nn.MSELoss()
    policy_criterion = CrossEntropy
    optim = Optimizer.Adam(net.parameters(), weight_decay=.0001)

    for i in range(epochs):
        start_time = time.time()
        avg_loss = Epoch(net, data_loader, value_criterion,
                         policy_criterion, optim)
        epoch_time = time.time() - start_time
        print("Epoch time:", epoch_time, "loss:", avg_loss)
        if i % 10 == 0:
            torch.save(net.state_dict(), save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plays Go games to generate training data")

    parser.add_argument('output_file', type=str,
                        help=f"Where the model is stored")
    parser.add_argument('--data_file', type=str, default=None,
                        help=f"The initial data to train on")
    parser.add_argument('--init_model', type=str, default=None,
                        help="The model that training is initiated with")
    parser.add_argument('--server_config', type=str, default=None,
                        help="A json config file giving the IP address and port for this compute node for distributed training.")
    args = parser.parse_args()

    if args.data_file is not None:
        # Load the playouts stored, to be used as the initial training data
        playouts: List[List[MoveDatapoint]] = LoadObject(args.data_file)
    else:
        playouts = []

    # Load the replay memory
    replay_memory = ReplayMemory(16384)
    for playout in playouts:
        for move_datapoint in playout:
            replay_memory.push(move_datapoint)

    # Setup the network
    board_size = len(playouts[0][0].board_list)
    net = Resnet(2, board_size).to(DEVICE)
    if args.init_model is not None:
        net.Load(args.init_model)

    mem_lock = Lock()
    param_queue = None
    server = None
    shutdown_val = None
    receiver_proc = None
    if args.server_config is not None:
        # Setup the handling of workers and the parameter server
        with open(args.server_config) as server_file:
            config = json.load(server_file)

        if ('addr' not in config) or ('port' not in config):
            print("IP address (addr) and port number required in config")

        address = config['addr']
        port = int(config['port'])

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((address, port))
        server.listen()

        param_queue = Queue()
        param_queue.put(net.state_dict())

        shutdown_val = Value('b', 0)

        receiver_proc = Process(target=HandleWorkers,
                                args=(server, replay_memory, mem_lock,
                                      param_queue, shutdown_val))
        receiver_proc.start()

    while True:
        try:
            Train(net, replay_memory, mem_lock, args.output_file)
            if param_queue is not None:
                param_queue.put(net.state_dict)
            torch.save(net.state_dict(), args.output_file)
        except KeyboardInterrupt:
            if server is not None:
                assert(shutdown_val is not None and receiver_proc is not None)
                print("Shutting down...")

                with shutdown_val.get_lock():
                    shutdown_val.value = 1
                receiver_proc.join()
                server.close()
            break
