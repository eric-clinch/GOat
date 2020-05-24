
import math
import time
from py_mcts.Board import MCTSMove, Policy, Board
from py_mcts.MCTS import MoveInfo, Struct, TreeNode
import numpy as np


class BatchTreeNode(TreeNode):
    def __init__(self, board, parent):
        self.board = board
        self.parent = parent

        self.moves = []
        if board.IsGameOver():
            self.winner = board.GetWinner()
            self.value_sum = 1 if self.winner == board.current_player else 0
        else:
            self.winner = None
            self.value_sum = None
            if board.GameEffectivelyOver() or board.PassWins():
                pass_move = Struct()
                pass_move.row, pass_move.col = -1, -1
                self.moves = [pass_move]
            else:
                self.moves = board.GetLegalMoves()
            self.children = []

            assert(len(self.moves) > 0)
            self.NormalizeChildrenPriors()

        self.batch_visits = 0
        self.visits = 1

    def LazyEvaluate(self, value, board_prior, pass_prior):
        # Set the board to None so the memory can be freed now that it is no longer needed
        self.board = None

        if self.winner is not None:
            value = self.value_sum
        else:
            if self.value_sum is not None:
                # This node was visited multiple times, the visit is redundant
                return
            self.value_sum = value
            self.children = [MoveInfo(move, board_prior, pass_prior)
                             for move in self.moves]
        if self.parent is not None:
            self.parent.BackpropEvaluation(1 - value)

    def BackpropEvaluation(self, value):
        self.value_sum += value
        self.visits += 1
        self.batch_visits = 0

        if self.parent is not None:
            self.parent.BackpropEvaluation(1 - value)

    def PUCT(self):
        best_score = None
        best_index = None
        for i in range(len(self.children)):
            child = self.children[i]
            if child.node is None:
                child_score = 1
                child_visits = 0
                child_batch_visits = 0
            else:
                child_score = (1 - child.node.ExpectedValue()
                               if child.node.value_sum is not None else 0)
                child_visits = child.node.visits
                child_batch_visits = child.node.batch_visits

            exploration_weight = math.sqrt(2)
            child_score += exploration_weight * child.prior * \
                math.sqrt(math.log(self.visits) / (1 + child_visits))
            child_score -= child_batch_visits / math.sqrt(self.visits)
            if best_score is None or child_score > best_score:
                best_score = child_score
                best_index = i
        return best_index

    # Performs an iteration of MCTS, returning a node to be (lazily) evaluated
    def MCTSIter(self, board):
        res_node = None
        if self.winner is not None:
            # This is a terminal node, just return itself
            res_node = self
        else:
            choice_idx = self.PUCT()
            choice_child = self.children[choice_idx]
            choice_move = choice_child.move
            board.MakeMove(choice_move.row, choice_move.col)
            if choice_child.node is None:
                choice_child.node = BatchTreeNode(board, self)
                res_node = choice_child.node
            elif choice_child.node.value_sum is None:
                # This child has already been visited but not yet evaluated. Just return the child
                res_node = choice_child.node
            else:
                res_node = choice_child.node.MCTSIter(board)
        self.batch_visits += 1
        return res_node


def ProcessBatch(batch_nodes, batch_evaluator, board_size):
    batch_boards = [node.board if node.board is not None else Board(
        board_size) for node in batch_nodes]
    values, board_priors, pass_priors = batch_evaluator(batch_boards)
    assert(len(batch_nodes) == len(values))
    assert(len(batch_nodes) == len(board_priors))
    assert(len(batch_nodes) == len(pass_priors))
    for i in range(len(batch_nodes)):
        batch_nodes[i].LazyEvaluate(values[i], board_priors[i], pass_priors[i])


def BatchedMCTS(board, batch_evaluator, seconds_to_run):
    start_time = time.time()

    root_node = BatchTreeNode(board, None)
    root_batch = [root_node]
    ProcessBatch(root_batch, batch_evaluator, len(board))

    count = 0
    batch_size = 64
    while time.time() - start_time < seconds_to_run:
        batch_nodes = []
        for _ in range(batch_size):
            batch_nodes.append(root_node.MCTSIter(board.Copy()))
            count += 1
        ProcessBatch(batch_nodes, batch_evaluator, len(board))

    temp = .25
    move, confidence = root_node.SampleChild(temp=temp)

    print("%d Batch MCTS iterations performed" % count)

    result = Struct()
    if confidence < 0.05:
        # pass
        result.row = -1
        result.col = -1
    else:
        result.row = move.row
        result.col = move.col
    result.confidence = confidence
    result.policy = root_node.VisitPolicy(len(board), temp=temp)

    return result
