
py_binary(
    name = "ui",
    srcs = ["ui.py"],
    deps = ["//code/py_alphago/py_mcts:py_mcts",
            "//code/py_alphago/resnet:resnet"],
)

py_binary(
    name = "play_games",
    srcs = ["play_games.py", "communication.py"],
    deps = ["//code/py_alphago/py_mcts:py_mcts",
            "//code/py_alphago/resnet:resnet"],
)

py_binary(
    name = "train_network",
    srcs = ["train_network.py", "play_games.py",
            "communication.py"],
    deps = ["//code/py_alphago/py_mcts:py_mcts",
            "//code/py_alphago/resnet:resnet"],
)
