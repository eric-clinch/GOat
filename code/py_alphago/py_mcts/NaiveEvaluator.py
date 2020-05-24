
def NaiveEvaluator(board):
  prior = [[1] * len(board) for _ in range(len(board))]
  pass_prior = 0
  value = board.RandomPlayout() 
  return value, prior, pass_prior
