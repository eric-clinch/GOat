#ifndef TREENODE_H__
#define TREENODE_H__

#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "code/cpp_mcts/board/board.h"
#include "code/cpp_mcts/board/move.h"
#include "mab.h"
#include "tree_node.h"
#include "utility_node.h"

class TreeNode {
 public:
  TreeNode(const Board &board, Player playerID, Player enemyID);

  ~TreeNode();

  std::tuple<int, TreeNode *, bool> getAndMakeMove(
      const MAB<Move> &mab, Board &board,
      const std::unordered_map<Move, double> &searchPriorMap);

  void updateUtility(int moveIndex, float utility);

  const Move getMostVisited() const;
  float getConfidence() const;

  // Returns a flat probability distribution vector with an entry for each
  // possible move in a Go game of the given board size (includes both legal and
  // illegal moves). The size of the vector is then board_size * board_size + 1
  // (including the pass move)
  std::vector<float> getVisitDistribution(int board_size) const;

  bool isLeaf() const;

  size_t getNumMoves() const;

  std::unordered_map<Move, double> getSearchPriorMap() const;

  int getCount() {
    int s = 0;
    for (const UtilityNode<Move> &node : moveUtilities) {
      s += node.numTrials;
    }
    return s;
  }

 private:
  Player playerID;
  Player enemyID;

  std::vector<UtilityNode<Move>> moveUtilities;
  std::vector<unsigned int> moveThreadCounts;
  std::vector<TreeNode *> children;

  mutable std::mutex node_mtx;
  size_t visits;
};

#endif  // TREENODE_H__
