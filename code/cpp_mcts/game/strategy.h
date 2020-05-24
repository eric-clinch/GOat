#ifndef STRATEGY_h
#define STRATEGY_h

#include "cpp_mcts/board/board.h"
#include "cpp_mcts/board/move.h"

class Strategy {
 public:
  virtual ~Strategy() {}
  virtual const Move getMove(const Board &board, Player playerID,
                             Player enemyID) = 0;

  virtual string toString() { return "Strategy"; }
};

#endif
