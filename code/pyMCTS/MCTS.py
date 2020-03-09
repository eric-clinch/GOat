
import math
import time
from Board import MCTSMove

class MoveInfo():
  def __init__(self, move, board_prior):
    self.move = move
    self.node = None
    if move.row < 0:
      assert(move.col < 0)
      self.prior = 0
    else:
      self.prior = board_prior[move.row][move.col]

class TreeNode():
  def __init__(self, board, evaluator):
    if board.IsGameOver():
      self.winner = board.GetWinner()
      value = 1 if self.winner == board.current_player else 0
    else:
      self.winner = None
      moves = board.GetLegalMoves()
      assert(len(moves) > 0)
      value, board_prior = evaluator(board)
      self.children = [MoveInfo(move, board_prior) 
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

  def MCTSIter(self, board, evaluator):
    if self.winner is not None:
      value = 1 if self.winner == board.current_player else 0
    else:
      choice_idx = self.UCB1()
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

  def PrintTree(self, depth, max_depth):
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

  def ExpectedValue(self):
    return self.value_sum / self.visits

  def UCB1(self):
    best_score = None
    best_index = None
    for i in range(len(self.children)):    
      child = self.children[i]
      if child.node is None:
        return i
      child_score = 1 - child.node.ExpectedValue()
      child_score += math.sqrt(2 * math.log(self.visits) / child.node.visits)
      if best_score is None or child_score > best_score:
        best_score = child_score
        best_index = i
    return best_index

  # Returns the move and the expected value associated with the most visited
  # child
  def BestChild(self):
    most_visits = 0
    best_child = None
    for child in self.children:
      if child.node.visits > most_visits:
        most_visits = child.node.visits
        best_child = child
    return best_child.move, 1 - best_child.node.ExpectedValue()

def MCTS(board, evaluator, seconds_to_run):
  start_time = time.time()
  root_node = TreeNode(board, evaluator)
  while time.time() - start_time < seconds_to_run:
    root_node.MCTSIter(board.Copy(), evaluator)
  move, confidence = root_node.BestChild()

  # root_node.PrintTree(0, 3)

  result = MCTSMove()
  result.row = move.row
  result.col = move.col
  result.confidence = confidence

  return result
