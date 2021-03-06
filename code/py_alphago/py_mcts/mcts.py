
from typing import *

import math
import time
import numpy as np

from py_mcts.board import MCTSMove, Policy, Location, Board


class MoveInfo():
    def __init__(self, move: Location, board_prior: List[List[float]],
                 pass_prior: float):
        self.move = move
        self.node = None
        if move.row < 0:
            assert(move.col < 0)
            self.prior = pass_prior
        else:
            self.prior = board_prior[move.row][move.col]


class Struct():
    pass

# The evaluator used here is expected to be a function that takes a Board as
# input and returns a Tuple[float, List[List[float]], float], where the first
# tuple element is a float is an approximation of the game value, the second
# element is a heuristic prior over what move is best, and the final element
# is a float weight for the pass move. The heuristic prior and the pass move
# weight are used together to create a probability distribution over the
# possible moves for the agent to make to guide MCTS.

class TreeNode():
    def __init__(self, board: Board, evaluator):
        if board.IsGameOver():
            self.winner = board.GetWinner()
            value = 1 if self.winner == board.current_player else 0
        else:
            if board.GameEffectivelyOver() or board.PassWins():
                pass_move = Struct()
                pass_move.row, pass_move.col = -1, -1
                moves = [pass_move]
            else:
                moves = board.GetLegalMoves()

            self.winner = None
            assert(len(moves) > 0)
            value, board_prior, pass_prior = evaluator(board)
            self.children = [MoveInfo(move, board_prior, pass_prior)
                             for move in moves]
            self.NormalizeChildrenPriors()

        self.visits = 1
        self.value_sum = value

    def NormalizeChildrenPriors(self):
        prior_sum = 0
        for child in self.children:
            prior_sum += child.prior
        for child in self.children:
            child.prior /= prior_sum

    def MCTSIter(self, board: Board, evaluator) -> float:
        if self.winner is not None:
            value = 1 if self.winner == board.current_player else 0
        else:
            choice_idx = self.PUCT()
            # choice_idx = self.UCB1()
            choice_child = self.children[choice_idx]
            choice_move = choice_child.move
            board.MakeMove(choice_move.row, choice_move.col)
            if choice_child.node is None:
                choice_child.node = TreeNode(board, evaluator)
                value = 1 - choice_child.node.value_sum
            else:
                value = 1 - choice_child.node.MCTSIter(board, evaluator)
        self.visits += 1
        self.value_sum += value
        return value

    def PrintTree(self, depth: int, max_depth: int):
        if depth == max_depth:
            return
        print('\t' * depth, self.visits, self.ExpectedValue())
        if self.winner is not None:
            return
        for child in self.children:
            if depth + 1 < max_depth:
                print('\t' * (depth + 1), child.move.row, child.move.col)
                if child.node is not None:
                    child.node.PrintTree(depth + 1, max_depth)

    def ExpectedValue(self) -> float:
        return self.value_sum / self.visits

    # Returns the index of the move chosen by the UCB1 algorithm
    def UCB1(self) -> int:
        # If there are unvisited children, pick the child with the highest prior
        best_prior = None
        best_index = None
        for i in range(len(self.children)):
            child = self.children[i]
            if child.node is not None:
                continue
            if best_prior is None or child.prior > best_prior:
                best_index = i
                best_prior = child.prior

        if best_index is not None:
            return best_index

        # All the children have been visited at least once, so pick the one with the highest UCB1 score
        best_score = None
        for i in range(len(self.children)):
            child = self.children[i]
            child_score = 1 - child.node.ExpectedValue()
            child_score += math.sqrt(2 *
                                     math.log(self.visits) / child.node.visits)
            if best_score is None or child_score > best_score:
                best_score = child_score
                best_index = i
        return best_index

    # Returns the index of the move chosen by the UCB1 algorithm
    def PUCT(self) -> int:
        best_score = None
        best_index = None
        for i in range(len(self.children)):
            child = self.children[i]
            if child.node is None:
                child_score = 1
                child_visits = 0
            else:
                child_score = 1 - child.node.ExpectedValue()
                child_visits = child.node.visits

            exploration_weight = math.sqrt(2)
            child_score += exploration_weight * child.prior * \
                math.sqrt(math.log(self.visits) / (1 + child_visits))
            if best_score is None or child_score > best_score:
                best_score = child_score
                best_index = i
        return best_index

    # Returns the move and the expected value associated with the most visited
    # child
    def BestChild(self) -> Tuple[Location, float]:
        most_visits = 0
        best_child = None
        for child in self.children:
            if child.node is not None and child.node.visits > most_visits:
                most_visits = child.node.visits
                best_child = child
        return best_child.move, 1 - best_child.node.ExpectedValue()

    def VisitPolicy(self, board_size, temp=1) -> Policy:
        num_elems = board_size * board_size + 1
        distribution = [0] * num_elems
        weight_sum = 0
        for child in self.children:
            move = child.move
            idx = board_size * move.row + move.col
            if move.row < 0:
                assert(move.col < 0)
                # This is the pass move
                idx = num_elems - 1
            weight = (child.node.visits ** (1 / temp)
                      if child.node is not None else 0)
            distribution[idx] = weight
            weight_sum += weight

        # Normalize the distribution
        for i in range(num_elems):
            distribution[i] = distribution[i] / weight_sum

        # Because the Policy class inherits from ctypes.Structure, you can't
        # initiate it with Python values. Instead we'll just create a struct
        # and populate its attributes so that it matches the Policy class
        # interface.
        policy = Struct()
        policy.length = len(distribution)
        policy.distribution = distribution
        return policy

    # Samples a move. Each move is sampled with probability proportional to
    # v ** (1 / t), where v is the visit count of that move, and t is the
    # temperature.
    def SampleChild(self, temp=1) -> Tuple[Location, float]:
        child_weights = []
        for child in self.children:
            if child.node is None:
                child_weights.append(0)
            else:
                child_weights.append(child.node.visits ** (1 / temp))
        child_weights = np.array(child_weights)
        child_weights = child_weights / sum(child_weights)
        sample_child = np.random.choice(self.children, p=child_weights)
        return sample_child.move, 1 - sample_child.node.ExpectedValue()


def MCTS(board: Board, evaluator, seconds_to_run: float) -> MCTSMove:
    start_time = time.time()
    root_node = TreeNode(board, evaluator)
    count = 0
    while time.time() - start_time < seconds_to_run:
        root_node.MCTSIter(board.Copy(), evaluator)
        count += 1
    move, confidence = root_node.SampleChild()

    print(f"{count} MCTS iterations performed")

    # Because the MCTSMove class inherits from ctypes.Structure, you can't
    # initiate it with Python values. Instead we'll just create a struct
    # and populate its attributes so that it matches the MCTSMove class
    # interface.
    result = Struct()
    if confidence < 0.05:
        # pass
        result.row = -1
        result.col = -1
    else:
        result.row = move.row
        result.col = move.col
    result.confidence = confidence
    result.policy = root_node.VisitPolicy(len(board))

    return result
