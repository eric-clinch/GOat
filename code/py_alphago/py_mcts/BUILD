
py_library(
    name = "py_mcts",
    srcs = ["BatchedMCTS.py","Board.py", "MCTS.py", "NaiveEvaluator.py",
            "NNEvaluator.py",],
    data = ["//cpp_mcts:libboard.so"],
    visibility = ["//py_alphago:__subpackages__"]
)
