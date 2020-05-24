
py_library(
    name = "py_mcts",
    srcs = ["batched_mcts.py", "board.py", "mcts.py", "naive_evaluator.py",
            "nn_evaluator.py",],
    data = ["//cpp_mcts:libboard.so"],
    visibility = ["//py_alphago:__subpackages__"]
)
