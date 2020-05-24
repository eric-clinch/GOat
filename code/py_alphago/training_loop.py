from play_games import GeneratePlayouts, PyMctsFactory, MoveDatapoint
from train_network import Train, LoadObject
import argparse
from resnet.resnet import Resnet, DEVICE
from py_mcts.NNEvaluator import NNEvaluatorFactory
import pickle, os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs the training loop")

    parser.add_argument('data_dir', type=str,
                        help="Directory where the playout data will be stored")
    parser.add_argument('network_dir', type=str, 
                    help="Directory where the trained networks will be stored")
    parser.add_argument('init_data', type=str,
                        help="The playout data that training will start with")
    parser.add_argument('--init_model', type=str, default=None,
                        help="The model that training is initiated with")
    parser.add_argument('--games', type=int, default=200,
                        help="The number of games to be played per loop iteration")
    parser.add_argument('--move_time', type=int, default=10,
                        help="The seconds of search per move")
    parser.add_argument('--train_epochs', type=int, default=500,
                        help="The number of optimization epochs per loop iteration")

    args = parser.parse_args()

    playouts = LoadObject(args.init_data)

    board_size = len(playouts[0][0].board_list)
    net = Resnet(2, board_size).to(DEVICE)
    if args.init_model is not None:
        net.Load(args.init_model)

    curr_iter = 0
    while True:
        network_filename = "iter_%d" % curr_iter
        network_save_path = args.network_dir + os.sep + network_filename
        Train(playouts, net, network_save_path, epochs=args.train_epochs)

        data_filename = "playouts_%d" % curr_iter
        data_save_path = args.data_dir + os.sep + data_filename
        evaluator = NNEvaluatorFactory(network_save_path, board_size)
        strategy = PyMctsFactory(evaluator, args.move_time)
        GeneratePlayouts(board_size, strategy, data_save_path, args.games)

        playouts = LoadObject(data_save_path)

        curr_iter += 1
        print("-" * 20, curr_iter, "iterations completed", '-' *20)
