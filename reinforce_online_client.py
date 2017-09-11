import generals
import torch
from torch.autograd import Variable
import ActorCritic
import numpy as np
import time
import argparse


def gen_state(update):
    label_map = np.array(update['tile_grid'])
    army_map = np.array(update['army_grid'])

    armies = update['armies']
    cities = update['cities']
    # Model was trained on turns offset by 1
    turn_num = update['turn'] - 1
    index = update['player_index']
    general_list = update['generals']

    op_index = 1 - index
    state = np.zeros((11, label_map.shape[0], label_map.shape[1]))

    """Current Schema for game 1v1 game state:
            channel 0: army values of user
            channel 1: observed army values of opponent
            channel 2: binary values indicating obstacle
            channel 3: army values of observed neutral cities
            channel 4: coordinates of observed mountains
            channel 5: coordinates with values of capital
            channel 6: 1s where location is unobserved
            channel 7: coordinates of cities owned by self
            channel 8: coordinates of cities owned by enemy player
            channel 9: turn number % 50
            channel 10: enemy soldier number relative to own"""

    label_mask = label_map == index
    op_mask = label_map == op_index
    state[0][label_mask] = army_map[label_mask]
    state[1][op_mask] = army_map[op_mask]

    mountain_mask = (label_map == generals.MOUNTAIN)
    ob_mask = (label_map == generals.OBSTACLE)
    fog_mask = (label_map == generals.FOG)
    block_mask = mountain_mask + ob_mask
    unob_mask = fog_mask + ob_mask

    state[2][block_mask] = 1
    state[4][mountain_mask] = 1
    state[6][unob_mask] = 1
    state[9] = turn_num / 50.
    state[10] = min(armies[index] / 1. / max(armies[op_index], 1), 10.) / 10.

    for city in cities:
        if label_mask[city] == op_index:
            state[8][city] = army_map[city]
        elif label_mask[city] == index:
            state[7][city] = army_map[city]
        else:
            state[3][city] = army_map[city]

    for general in general_list:
        if general[0] >= 0:
            state[5][general] = army_map[general]

    return state[np.newaxis, ...]


def gen_valid_move(move_index, label_map, army_map, dims):
    """Generate the top valid move given an output from network"""
    x1, y1, x2, y2 = 0, 0, 0, 0
    move_half = False

    for i in range(moves.shape[0]):
        move = moves[i]
        if action_mask[move] == 0:
            break

        move_type, y1, x1 = np.unravel_index(move, (8, dims[0], dims[1]))
        index = move_type % 4

        if index == 0:
            x2, y2 = x1, y1 + 1
        elif index == 1:
            x2, y2 = x1 + 1, y1
        elif index == 2:
            x2, y2 = x1, y1 - 1
        elif index == 3:
            x2, y2 = x1 - 1, y1

        move_half = True if move_type >= 4 else False

        if y2 < 0 or y2 >= dims[0] or x2 < 0 or x2 >= dims[1]:
            continue

        if not (
            label_map[
                y2,
                x2] == generals.MOUNTAIN) and (
            army_map[
                y1,
                x1] > 1):
            break

    return x1, y1, x2, y2, move_half


if __name__ == "__main__":
    model = ActorCritic.ActorCritic()
    model.load_state_dict(torch.load("reinforce.mdl"))
    model = model.eval()
    init_state = False

    parser = argparse.ArgumentParser(description='Policy Bot Player')
    parser.add_argument('--user_id', type=str, default="5900688366",
                        help='user_id for bot')
    parser.add_argument('--username', type=str, default="[Bot] asdfshqwen123",
                        help='username for bot')
    parser.add_argument('--game_id', type=str, default="viz0",
                        help='id for the game')
    args = parser.parse_args()

    # private game
    g = generals.Generals(args.user_id, args.username, 'private', args.game_id)

    for update in g.get_updates():
        start_time = time.time()
        state = gen_state(update)
        dims = state.shape[2], state.shape[3]

        label_map = np.array(update['tile_grid'])
        army_map = np.array(update['army_grid'])

        army_map = state[0, 0, ...]
        label_mask = army_map > 0
        full_label_mask = np.concatenate(
            [label_mask[np.newaxis, ...] for i in range(8)])

        if not init_state:
            model.init_hidden(*dims)
            init_state = True

        val, action = model.forward(Variable(torch.Tensor(state)))
        action = np.e ** action
        action_mask = action.data.numpy().squeeze() * full_label_mask.flat
        moves = action_mask.argsort()[::-1]
        move = torch.from_numpy(action_mask).multinomial(1).numpy().flat[0]

        move_type, y1, x1 = np.unravel_index(move, (8, dims[0], dims[1]))
        index = move_type % 4

        if index == 0:
            x2, y2 = x1, y1 + 1
        elif index == 1:
            x2, y2 = x1 + 1, y1
        elif index == 2:
            x2, y2 = x1, y1 - 1
        elif index == 3:
            x2, y2 = x1 - 1, y1

        move_half = True if move_type >= 4 else False

        # Uncomment if want to generate next valid move from reinforce bot
        # instead of sampling a probability
        # x1, y1, x2, y2, move_half = gen_valid_move(
        #     moves, label_map, army_map, dims)

        print(x1, y1, x2, y2)
        g.move(y1, x1, y2, x2, move_half=move_half)
        print("--- {} seconds --- in turn {}".format((time.time() -
                                                      start_time), update['turn']))
