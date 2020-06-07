import py_mcts
from py_mcts.mcts import MCTS
from py_mcts.board import Board
from py_mcts.naive_evaluator import NaiveEvaluator
from py_mcts.nn_evaluator import NNEvaluatorFactory
from torch.multiprocessing import Process, Queue
from typing import *

import pickle
import argparse
import json
import socket
import pickle
import io
import string
import torch


class MoveDatapoint():
    def __init__(self, board_list: List[List[str]], player: int, confidence: float,
                 policy: py_mcts.board.Policy):
        self.board_list = board_list
        self.player = player
        self.confidence = confidence
        self.policy: List[float] = [policy.distribution[i]
                                    for i in range(policy.length)]


# -----------------------------------------------------------
# Socket communication code
# -----------------------------------------------------------


# A byte delimiter that is very unlikely to occur naturally
DELIM = string.printable.encode('UTF-8')


# Sends the given message with the delimiter
def Send(sckt: socket.socket, msg):
    sckt.send(msg + DELIM)


# Given a byte string that ends with the delimiter, returns the byte string
# with the delimiter stripped from the end
def StripDelim(msg):
    assert(len(msg) >= len(DELIM))
    return msg[:-len(DELIM)]


# Given a training server socket, listens on that socket for updates to the
# network and places them in the param queue.
def ReceiveParams(server: socket.socket, param_queue: Queue):
    server.setblocking(True)
    print("Listening for network updates...")
    msg = b''
    while True:
        try:
            msg += server.recv(8192)
        except Exception as err:
            print(f"Error with server connection, ending connection")
            server.close()
            return

        if msg.endswith(DELIM):
            msg = StripDelim(msg)

            buffer = io.BytesIO(msg)
            state_dict = torch.load(buffer)

            param_queue.put(state_dict)

            msg = b''


def SendPlayout(training_server: socket.socket, playout: List[MoveDatapoint]):
    msg = pickle.dumps(playout)
    Send(training_server, msg)
    print(f"Sent playout, message size {len(msg)}")


# -----------------------------------------------------------
# Playout generation code
# -----------------------------------------------------------


def GeneratePlayout(board_size: int, strategy0, strategy1) -> List[MoveDatapoint]:
    board = Board(board_size)
    winner = 0
    move_datapoints = []
    while not board.IsGameOver() and len(move_datapoints) < (board_size * board_size):
        strategy = strategy0 if board.current_player == 0 else strategy1
        mcts_move: py_mcts.board.MCTSMove = strategy(board)
        move_row, move_col = mcts_move.row, mcts_move.col
        if mcts_move.confidence < .1:
            winner = 1 - board.current_player
            break  # Concede the game

        move_datapoints.append(MoveDatapoint(board.BoardList(),
                                             board.current_player,
                                             mcts_move.confidence,
                                             mcts_move.policy))

        board.MakeMove(move_row, move_col)

    if board.IsGameOver():
        winner = board.GetWinner()

    return move_datapoints


def CppMctsFactory(threads, seconds):
    def strategy(board):
        return board.GetMCTSMove(threads, seconds)
    return strategy


def PyMctsFactory(evaluator, seconds):
    def strategy(board):
        return MCTS(board, evaluator, seconds)
    return strategy


def WriteObject(file_name: str, obj):
    file = open(file_name, 'wb')
    pickle.dump(obj, file)
    file.close()


def LoadObject(file_name: str):
    file = open(file_name, 'rb')
    obj = pickle.load(file)
    file.close()
    return obj


# Runs self-play games using the given strategy, and saves the playout data
# from these games in the given save path. The data is stored using pickle as
# a List[List[MoveDatapoint]] object. Here the each element of the outer list
# is a game playout, and the elements of the inner lists are the data for
# each move in the game.
def GeneratePlayouts(board_size: int, strategy, save_path: Optional[str],
                     playouts: List[MoveDatapoint],
                     training_server: Optional[socket.socket],
                     param_queue: Optional[Queue]):
    game_count = 0
    while True:
        playout = GeneratePlayout(board_size, strategy, strategy)
        print(game_count + 1, "games played")
        game_count += 1

        if save_path is not None:
            playouts.append(playout)
            WriteObject(save_path, playouts)

        if training_server is not None:
            SendPlayout(training_server, playout)

        if param_queue is not None and not param_queue.empty():
            print("Received new parameters from the training server")
            while not param_queue.empty():
                state_dict = param_queue.get()
            evaluator = NNEvaluatorFactory(state_dict, board_size)
            strategy = PyMctsFactory(evaluator, seconds_per_move)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plays Go games to generate training data")

    parser.add_argument('--model', type=str, default=None,
                        help="The model to use to play the game. If not given, naive MCTS will be used.")
    parser.add_argument('--output_file', type=str, default=None,
                        help="Where to store the data")
    parser.add_argument('--append', default=False, action='store_true',
                        help="Appends the generated data to the given output file rather than rewriting it")
    parser.add_argument('--server_config', type=str, default=None,
                        help="A json config file giving the IP address and port for the parameter training server. If this is given, the model parameter will be ignored")
    args = parser.parse_args()

    board_size = 9
    seconds_per_move = 5
    training_server = None
    param_queue = None

    if args.server_config is not None:
        with open(args.server_config) as server_file:
            config = json.load(server_file)

        if ('addr' not in config) or ('port' not in config):
            print("IP address (addr) and port number required in config")

        address = config['addr']
        port = int(config['port'])

        training_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        training_server.connect((address, port))
        param_queue = Queue()

        receiver_proc = Process(target=ReceiveParams,
                                args=(training_server, param_queue),
                                daemon=True)
        receiver_proc.start()

        # Wait for the training server to send us the initial model parameters
        state_dict = param_queue.get()
        evaluator = NNEvaluatorFactory(state_dict, board_size)
        strategy = PyMctsFactory(evaluator, seconds_per_move)
    elif args.model is not None:
        evaluator = NNEvaluatorFactory(args.model, board_size)
        strategy = PyMctsFactory(evaluator, seconds_per_move)
    else:
        strategy = CppMctsFactory(-1, seconds_per_move)

    playouts = []
    if args.output_file is not None and args.append:
        playouts = LoadObject(args.output_file)

    GeneratePlayouts(board_size, strategy, args.output_file, playouts,
                     training_server, param_queue)
