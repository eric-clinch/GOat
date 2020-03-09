
def NaiveEvaluator(board):
  prior = [[1] * len(board) for _ in range(len(board))]
  value = board.RandomPlayout() 
  return value, prior
