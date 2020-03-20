from Resnet.resnet import Resnet
import torch
import torch.optim as optimizer
import pickle
from PlayGames import MoveDatapoint

import time, math, random
import argparse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def LoadObject(file_name):
    file = open(file_name, 'rb')
    obj = pickle.load(file)
    file.close()
    return obj

def GetPlayerStoneMap(board_list, player):
    player_str = 'B' if player == 0 else 'W'
    stone_mapper = lambda x: 1 if x == player_str else 0
    stone_map = [list(map(stone_mapper, row)) for row in board_list]
    return stone_map

def GetDataset(playouts, batchsize):
    datapoints = []
    for playout in playouts:
        for move_datapoint in playout:
            player = move_datapoint.player
            player_stone_map = GetPlayerStoneMap(move_datapoint.board_list, player)
            opponent_stone_map = GetPlayerStoneMap(move_datapoint.board_list, 1 - player)
            network_input = torch.Tensor([player_stone_map, opponent_stone_map])
            value = torch.Tensor([move_datapoint.confidence])
            policy = torch.Tensor(move_datapoint.policy)
            datapoints.append((network_input, value, policy))
    return torch.utils.data.DataLoader(datapoints, batch_size=batchsize, shuffle=True)

def AugmentData(states, policies):
    board_moves = policies[:, :-1]
    pass_probs = policies[:, -1].unsqueeze(1)
    shape = board_moves.shape
    batchsize = shape[0]
    board_size = int(math.sqrt(shape[1]))
    board_moves = board_moves.reshape([batchsize, board_size, board_size])

    if random.randint(0, 1) == 1:
        # flip the tensors
        states = states.flip([-2, -1])
        board_moves = board_moves.flip([-2, -1])
    rotations = random.randint(0, 3)
    for i in range(rotations):
        states = states.rot90(1, [-2, -1])
        board_moves = board_moves.rot90(1, [-2, -1])

    board_moves = board_moves.flatten(start_dim=1)
    policies = torch.cat([board_moves, pass_probs], 1)

    return states, policies 

def CrossEntropy(target, distribution):
    losses = torch.einsum('ij,ij->i', target, torch.log(distribution))
    return -torch.mean(losses)

def Epoch(nn, data_loader, value_criterion, policy_criterion, optim):
    running_loss = 0
    count = 0
    for batch in data_loader:
        optim.zero_grad()
        states, target_values, target_policies = batch
        states, target_policies = AugmentData(states, target_policies)

        states = states.to(DEVICE)
        target_values = target_values.to(DEVICE)
        target_policies = target_policies.to(DEVICE)
        nn_values, nn_policies = nn(states)
        value_loss = value_criterion(nn_values, target_values)
        policy_loss = policy_criterion(target_policies, nn_policies)
        loss = value_loss + policy_loss
        loss.backward()
        optim.step()

        running_loss += loss.item()
        count += 1

    return running_loss / count 

def Train(playouts, net, save_path, epochs=500):
    batch_size = 1024
    data_loader = GetDataset(playouts, batch_size)

    value_criterion = torch.nn.MSELoss()
    policy_criterion = CrossEntropy
    optim = optimizer.Adam(net.parameters(), weight_decay=.0001)

    for i in range(epochs):
        start_time = time.time()
        avg_loss = Epoch(net, data_loader, value_criterion, policy_criterion, optim)
        epoch_time = time.time() - start_time
        print("Epoch time:", epoch_time, "loss:", avg_loss)
        if i % 10 == 0:
            torch.save(net.state_dict(), save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plays Go games to generate training data")

    parser.add_argument('output_file', type=str, help=f"Where the model is stored")
    parser.add_argument('data_file', type=str, help=f"The data to train on")
    parser.add_argument('--init_model', type=str, default=None,
                        help="The model that training is initiated with")
    args = parser.parse_args()

    playouts = LoadObject(args.data_file)
    print('Training on %d playouts' % len(playouts))

    board_size = len(playouts[0][0].board_list)
    net = Resnet(2, board_size).to(DEVICE)
    if args.init_model is not None:
        net.Load(args.init_model)

    Train(playouts, net, args.output_file)
