# GOat
An implementation of AlphaGo. The training is done in a distributed fashion over socket communication, and a tkinter UI is provided to play against the trained agent.


### Requirements
- Bazel >=2.0.0
- Python >=3.8
- python3 tkinter (for UI only)

The Python modules required for this project can be seen in requirements.txt, and installed by running 

```pip install -r requirements.txt```

### How to train
The training is done in a distributed fashion. Worker nodes produce training data through self-play and send this data to a training node. The training node receives the self-play data, uses it to train the neural network, and periodically sends the parameters of the neural network to the workers.

To build the training and worker scripts, run
```
bazel build code/py_alphago/train_network
bazel build code/py_alphago/play_games
```

Both the training node and the worker nodes require a config file to run, which gives the IP address for the training node and the port that it will be communicating on. This config file should be supplied as a json file with the following format:
```
{
  "addr": "[training node IP address]",
  "port": [integer port for communication]
}
```

The training node can be started with
```
./bazel-bin/code/py_alphago/train_network [output file for trained model] --server_config config_file.json
```

A worker node can be started with
```
./bazel-bin/code/py_alphago/train_network --server_config config_file.json
```

The worker script also has options for writing the game data collected to a file (serialized using the pickle module), in place of or in addition to sending the data to the training server. The training server has options for using such a pickled data file to initially fill the replay memory, and also has an option for using a previously trained model as the starting point for the training. The scripts have ```--help``` flags for more information on these options. Some previously trained models and generated game data are available in the ```trained_models``` and ```data``` directories, respectively.

### How to play
A UI, written in tkinter, is available for playing against a trained model. To build this UI, run
```
bazel build code/py_alphago/user_ui
```

To use the UI, run
```
./bazel-bin/code/py_alphago/user_ui
```
Currently the UI is hard-coded to use the model at ```trained_models/naive_mcts_mode```. In the future, options will be added to allow the user to select what model to play against, and to let the user just watch the AI play against itself.

### Code architecture
This is meant to give an overview of the code structure for anyone that wants to read and or modify the code. All code is naturally located in the ```code``` directory. Inside there are two directories, ```cpp_mcts``` and ```py_alphago```.

This code base started as a final project for a class, with that project being an implementation of a parallelized version of MCTS to play Go. This project was all in C++ and lives in the ```cpp_mcts``` directory. There you can see the implementation of Go, a parallelized implementation of MCTS, and the code that allows a user to play against the AI. This code can be compiled into a shared object file to provide an interface for Python, allowing the Python side of this project to use the C++ implementation of Go and MCTS. Note that the C++ side and the UI were written over a year before the AlphaGo side of this project was started, so you might notice a difference in naming convention and style between the C++ and Python side of the code.

The bulk of the AlphaGo implementation is in the ```py_alphago``` directory. At the top level of this directory is the scripts for training the network and running the UI. The network architecture can be seen in the ```resnet``` directory. In the ```py_mcts``` directory is an implementation of MCTS that takes an evaluation function as input. Given a state of the game board, this evaluation function is expected to return a heuristic approximation for the value of the game, as well as a probability distribution over the possible moves on the board. This allows the MCTS algorithm to easily be used for either naive MCTS or the AlphaGo algorithm. To implement the naive MCTS algorithm, this evaluation function would use a random playout to approximate the value of the game, and would return the uniform distribution over the possible moves. To implement AlphaGo, a neural network is used to provide the approximation of the game value and the distribution over future moves. A batched implementation of MCTS is also available, where the node expansion and result backpropagation phases in MCTS are done lazily, allowing nodes to be evaluated in a batched manner by the neural network. This allows the user to increase the batchsize to trade off exploration intelligence for the increased efficiency of batched computation. Using a batchsize of 1 is equivalent to normal MCTS.
