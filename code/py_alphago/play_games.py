from py_mcts.mcts import MCTS
from py_mcts.board import Board
from py_mcts.naive_evaluator import NaiveEvaluator
from py_mcts.nn_evaluator import NNEvaluatorFactory

import pickle
import argparse

class MoveDatapoint():
    def __init__(self, board_list, player, confidence, policy):
        self.board_list = board_list
        self.player = player
        self.confidence = confidence
        self.policy = [policy.distribution[i] for i in range(policy.length)]

def GeneratePlayout(board_size, strategy0, strategy1):
    board = Board(board_size)
    winner = 0
    move_datapoints = []
    while not board.IsGameOver() and len(move_datapoints) < (board_size * board_size):
        strategy = strategy0 if board.current_player == 0 else strategy1
        mcts_move = strategy(board)
        move_row, move_col = mcts_move.row, mcts_move.col
        if mcts_move.confidence < .1:
            winner = 1 - board.current_player
            break # Concede the game

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

def WriteObject(file_name, obj):
    file = open(file_name, 'wb')
    pickle.dump(obj, file)
    file.close()

def LoadObject(file_name):
    file = open(file_name, 'rb')
    obj = pickle.load(file)
    file.close()
    return obj

def GeneratePlayouts(board_size, strategy, save_path, num_playouts,
                     playouts=None):
    if playouts is None:
        playouts = []
    for i in range(num_playouts):
        playouts.append(GeneratePlayout(board_size, strategy, strategy))
        WriteObject(save_path, playouts)
        print(i+1, "games played")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plays Go games to generate training data")

    parser.add_argument('output_file', type=str, nargs="?", help=f"Where the data is stored")
    parser.add_argument('--append', default=False, action='store_true',
                        help="Appends the generated data to the given output file rather than rewriting it")
    parser.add_argument('--games', type=int, default=1000, help="The number of games to be played")
    parser.add_argument('--model', type=str, default=None,
                        help="The model to use to play the game. If not given, naive MCTS will be used")
    args = parser.parse_args()

    board_size = 9
    if args.model is None:
        strategy = CppMctsFactory(-1, 5)
    else:
        evaluator = NNEvaluatorFactory(args.model, board_size)
        strategy = PyMctsFactory(evaluator, 5)

    playouts = []
    if args.append:
        playouts = LoadObject(args.output_file)

    if args.output_file is None:
        print("output file required")
        exit()

    GeneratePlayouts(board_size, strategy, args.output_file, args.games,
                     playouts)
