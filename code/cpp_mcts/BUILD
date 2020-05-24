
cc_binary(
    name = "libboard.so",
    srcs = ["board_py_interface.cpp",],
    deps = ["//cpp_mcts/board:board", "//cpp_mcts/game:game",
            "//cpp_mcts/mcts:mcts"],
    linkopts = ["-pthreads"],
    linkshared=True,
    visibility = ["//py_alphago:__subpackages__"],
)
