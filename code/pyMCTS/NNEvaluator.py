
from Resnet.resnet import Resnet
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def GetPlayerStoneMap(board_list, player):
    player_str = 'B' if player == 0 else 'W'
    stone_mapper = lambda x: 1 if x == player_str else 0
    stone_map = [list(map(stone_mapper, row)) for row in board_list]
    return stone_map

def BoardToTensor(board):
    board_list = board.BoardList()
    player = board.current_player
    player_stone_map = GetPlayerStoneMap(board_list, player)
    opponent_stone_map = GetPlayerStoneMap(board_list, 1 - player)
    network_input = torch.Tensor([player_stone_map, opponent_stone_map])
    return network_input

def NNEvaluatorFactory(model_path, board_size):
    net = Resnet(2, board_size)
    net.Load(model_path)
    net.train(False)

    def evaluate(board):
        net_input = BoardToTensor(board).unsqueeze(0)
        value, policy = net(net_input)
        value = value[0][0].item()

        policy = torch.squeeze(policy)
        policy, pass_move_prob = policy[:-1], policy[-1].item()
        policy = policy.reshape([len(board), len(board)]).tolist()

        return value, policy, pass_move_prob
        
    
    return evaluate

