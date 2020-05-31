#ifndef MCTS_H__
#define MCTS_H__

#include <atomic>
#include <random>
#include "cpp_mcts/board/board.h"
#include "cpp_mcts/board/move.h"
#include "cpp_mcts/game/strategy.h"
#include "mab.h"
#include "tree_node.h"

struct workerArg;
struct mainArg;

class MCTS : public Strategy {
 public:
  MCTS(int64_t msPerMove, unsigned int playoutThreads,
       unsigned int iterationThreads, double explorationConstant,
       double playoutPercent);
  ~MCTS();
  virtual const Move getMove(const Board &board, Player playerID,
                             Player enemyID);
  virtual string toString();

  double getConfidence() const;
  std::vector<float> getVisitDistribution() const;

  static int playout(Board *board, Player playerID, Player enemyID,
                     double playoutPercent);

 private:
  // perform one iteration of the MCTS algorithm starting from the given node
  static void *getMoveHelper(void *arg);
  float MCTSIteration(Board &board, Player playerID, Player enemyID,
                      TreeNode &node, workerArg *groupInfo,
                      std::unordered_map<Move, double> &searchPriorMap);

  // static void *MCTSIterationWorker(void *args);
  float performPlayouts(Board &board, Player playerID, Player enemyID,
                        workerArg *groupInfo);
  static void *playoutWorker(void *arg);
  static const Move sampleMove(std::vector<Move> &moves);

 private:
  int64_t msPerMove;
  unsigned int playoutThreads;
  unsigned int iterationThreads;
  MAB<Move> *mab;
  double explorationConstant;
  double playoutPercent;
  double confidence;
  std::vector<float> visit_distribution;

  static std::random_device rd;
  static std::mt19937 rng;
  static std::uniform_real_distribution<> uni;
};

struct workerArg {
  Board *board;
  double playoutPercent;
  Player playerID;
  Player enemyID;
  size_t workers;

  volatile std::atomic<size_t> activeCount;
  volatile std::atomic<size_t> winCount;
  volatile char barrierFlag;  // sense-reversing flag
};

// for the purposes of iteration parallelization
struct mainArg {
  MCTS *mcts;
  const Board *board;
  Player playerID;
  Player enemyID;
  TreeNode *node;
  unsigned int workerThreads;
  unsigned int iterationThreads;
  int64_t ms;
  volatile std::atomic<size_t> iterations;
};

#endif  // MCTS_H__