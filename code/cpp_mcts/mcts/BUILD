
cc_library(
    name = "mcts",
    hdrs = ["mcts.h", "mab.h", "tree_node.h", "ucb1.h", "utility_node.h"],
    srcs = ["mcts.cpp", "tree_node.cpp"],
    deps = ["//code/cpp_mcts/board:board", "//code/cpp_mcts/game:game",
            "//code/cpp_mcts/library:tools",],
    visibility = ["//code/cpp_mcts:__subpackages__",],
)
