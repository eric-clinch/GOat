
cc_binary(
    name = "libboard.so",
    srcs = ["board_py_interface.cpp",],
    deps = ["//code/cpp_mcts/board:board", "//code/cpp_mcts/game:game",
            "//code/cpp_mcts/mcts:mcts"],
    linkopts = ["-pthreads"],
    linkshared=True,
    visibility = ["//code/py_alphago:__subpackages__"],
)
