
cc_library(
    name = "MCTS",
    hdrs = ["MCTS.h", "MAB.h", "TreeNode.h", "UCB1.h", "UtilityNode.h"],
    srcs = ["MCTS.cpp", "TreeNode.cpp"],
    deps = ["//cpp_mcts/Board:Board", "//cpp_mcts/Game:Game",
            "//cpp_mcts/library:Tools",],
    visibility = ["//cpp_mcts:__subpackages__",],
)
