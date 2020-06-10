
from typing import *

def NaiveEvaluator(board) -> Tuple[float, List[List[float]], float]:
  prior = [[1] * len(board) for _ in range(len(board))]
  pass_prior = 0
  value = board.RandomPlayout() 
  return value, prior, pass_prior
