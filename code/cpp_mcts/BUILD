
cc_binary(
    name = "libboard.so",
    srcs = ["BoardPyInterface.cpp",],
    deps = ["//cpp_mcts/Board:Board", "//cpp_mcts/Game:Game",
            "//cpp_mcts/MCTS:MCTS"],
    linkopts = ["-pthreads"],
    linkshared=True,
    visibility = ["//py_alphago:__subpackages__"],
)
