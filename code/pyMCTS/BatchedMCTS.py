
import math, time
from pyMCTS.Board import MCTSMove, Policy
from MCTS import MoveInfo, Struct, TreeNode

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
      if board.GameEffectivelyOver() or board.PassWins():
        pass_move = Struct()
        pass_move.row, pass_move.col = -1, -1
        self.moves = [pass_move]
      else:
        self.moves = board.GetLegalMoves()
    self.children = []
    self.unexpanded_subtree_leafs = 0
    self.batch_visits = 0

    assert(len(self.moves) > 0)
    self.NormalizeChildrenPriors()

    self.visits = 1

  def LazyEvaluate(self, value, board_prior, pass_prior):
    if self.winner is not None:
      value = 1 if self.winner == board.current_player else 0
    else:
      self.value_sum = value
      self.children = [MoveInfo(move, board_prior, pass_prior) 
                        for move in self.moves]
    if self.parent is not None:
      self.parent.BackpropEvaluation(1 - value, len(self.children))

  def BackpropEvaluation(self, value, num_new_leafs):
    self.value_sum += value

    # subtract one for the leaf that was just expanded
    self.unexpanded_subtree_leafs -= 1    
    self.unexpanded_subtree_leafs += num_new_leafs
    
    if self.parent is not None:
      self.parent.BackpropEvaluation(1 - value, num_new_leafs)

  def MCTSIter(self, board):
    if self.winner is not None:
      value = 
