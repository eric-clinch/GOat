
from pyMCTS.MCTS import MCTS
from pyMCTS.Board import Board
from pyMCTS.NaiveEvaluator import NaiveEvaluator

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plays Go games to generate training data")

    parser.add_argument('output_file', type=str, nargs="?", help=f"Where the data is stored")
    parser.add_argument('--append', default=False, action='store_true',
                        help="Appends the generated data to the given output file rather than rewriting it")
    parser.add_argument('--games', type=int, default=1000, help="The number of games to be played")
    parser.add_argument('--model', type=str, default=None,
                        help="The model to use to play the game. If not given, naive MCTS will be used")
    args = parser.parse_args()

    cpp_mcts = CppMctsFactory(-1, 5)
    py_mcts = PyMctsFactory(NaiveEvaluator, 5)
    board_size = 9

    playouts = []
    if args.append:
        playouts = LoadObject(args.output_file)

    for i in range(args.games):
        playouts.append(GeneratePlayout(board_size, cpp_mcts, cpp_mcts))
        WriteObject(args.output_file, playouts)
        print(i+1, "games played")
