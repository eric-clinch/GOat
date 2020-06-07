
from resnet.resnet import Resnet, DEVICE
from typing import *

import torch
import collections


def GetPlayerStoneMap(board_list, player):
    player_str = 'B' if player == 0 else 'W'
    def stone_mapper(x): return 1 if x == player_str else 0
    stone_map = [list(map(stone_mapper, row)) for row in board_list]
    return stone_map


def BoardToTensor(board):
    board_list = board.BoardList()
    player = board.current_player
    player_stone_map = GetPlayerStoneMap(board_list, player)
    opponent_stone_map = GetPlayerStoneMap(board_list, 1 - player)
    network_input = torch.Tensor([player_stone_map, opponent_stone_map])
    return network_input


def NNEvaluatorFactory(model_params: Union[str, collections.OrderedDict], board_size: int):
    net = Resnet(2, board_size).to(DEVICE)
    if isinstance(model_params, str):
        # Load the params from a file
        net.Load(model_params)
    elif isinstance(model_params, collections.OrderedDict):
        # Load the params from this state dict
        net.load_state_dict(model_params)
    else:
        raise Exception("Invalid model parameter object")
    net.train(False)

    def evaluate(board):
        net_input = BoardToTensor(board).unsqueeze(0).to(DEVICE)
        value, policy = net(net_input)
        value = value[0][0].item()

        policy = torch.squeeze(policy)
        policy, pass_move_prob = policy[:-1], policy[-1].item()
        policy = policy.reshape([len(board), len(board)]).tolist()

        return value, policy, pass_move_prob

    return evaluate


def BatchNNEvaluatorFactory(model_path, board_size):
    net = Resnet(2, board_size).to(DEVICE)
    net.Load(model_path)
    net.train(False)

    def evaluate(board_batch):
        board_size = len(board_batch[0])
        net_input = [BoardToTensor(board) for board in board_batch]
        net_input = torch.stack(net_input, axis=0)
        net_input = net_input.to(DEVICE)
        values, policies = net(net_input)

        values = values.squeeze(axis=1).tolist()

        board_policies = policies[:, :-1]
        board_policies = [policy.reshape(
            [board_size, board_size]).tolist() for policy in board_policies]

        pass_probs = policies[:, -1]
        pass_probs = pass_probs.tolist()

        return values, board_policies, pass_probs

    return evaluate
