
py_library(
    name = "py_mcts",
    srcs = ["batched_mcts.py", "board.py", "mcts.py", "naive_evaluator.py",
            "nn_evaluator.py",],
    data = ["//code/cpp_mcts:libboard.so"],
    visibility = ["//code/py_alphago:__subpackages__"]
)
