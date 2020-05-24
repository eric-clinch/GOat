
cc_library(
    name = "mcts",
    hdrs = ["mcts.h", "mab.h", "tree_node.h", "ucb1.h", "utility_node.h"],
    srcs = ["mcts.cpp", "tree_node.cpp"],
    deps = ["//cpp_mcts/board:board", "//cpp_mcts/game:game",
            "//cpp_mcts/library:tools",],
    visibility = ["//cpp_mcts:__subpackages__",],
)
