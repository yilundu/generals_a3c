import generals
import torch
from torch.autograd import Variable
import CNNLSTMPolicy
import numpy as np
import time

model = CNNLSTMPolicy.CNNLSTMPolicy()
model.load_state_dict(torch.load("2_epoch.mdl"))
model = model.eval()
init_state = False

user_id = "5900688366"
username = "[Bot] asdfshqwen123"


def gen_state(update):
    label_map = np.array(update['tile_grid'])
    army_map = np.array(update['army_grid'])

    print(label_map)
    print(army_map)

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
    state[10] = min(armies[0] / 1. / armies[1], 10.) / 10.

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


def gen_move_pred_start(pred_start, pred_end):
    _, row, col = pred_start.shape
    start = pred_start.argmax()
    y1, x1 = start // col, start % col

    max_prob = -1 * float('inf')
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        x_new = x1 + dx
        y_new = y1 + dy

        if x_new < 0 or x_new >= col or y_new < 0 or y_new >= row:
            continue

        if pred_end[0][y_new][x_new] > max_prob:
            move_half = False
            x2, y2 = x_new, y_new
            max_prob = pred_end[0][y_new][x_new]

        if pred_end[1][y_new][x_new] > max_prob:
            move_half = True
            x2, y2 = x_new, y_new
            max_prob = pred_end[1][y_new][x_new]

    return x1, y1, x2, y2, move_half

def gen_move_max(pred_start, pred_end, label_map, index):
    label_map = np.array(label_map)
    max_prob = -1 * float('inf')
    row = pred_start.shape[1]
    col = pred_start.shape[2]

    for y in range(row):
        for x in range(col):
            if label_map[y, x] != index:
                continue

            start_prob = pred_start[0, y, x]
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                x_new = x + dx
                y_new = y + dy

                if x_new < 0 or x_new >= col or y_new < 0 or y_new >= row:
                    continue

                if label_map[y_new, x_new] == generals.MOUNTAIN:
                    continue

                if pred_end[0][y_new][x_new] + start_prob > max_prob:
                    move_half = False
                    x1, y1 = x, y
                    x2, y2 = x_new, y_new
                    max_prob = pred_end[0][y_new][x_new] + start_prob

                if pred_end[1][y_new][x_new]+ start_prob > max_prob:
                    move_half = True
                    x1, y1 = x, y
                    x2, y2 = x_new, y_new
                    max_prob = pred_end[1][y_new][x_new] + start_prob

    return x1, y1, x2, y2, move_half


# private game
g = generals.Generals(user_id, username, 'private', 'viz0')

for update in g.get_updates():
    start_time = time.time()
    state = gen_state(update)
    dims = state.shape[2], state.shape[3]

    if not init_state:
        model.init_hidden(*dims)
        init_state = True

    pred_s, pred_e = model.forward(Variable(torch.Tensor(state)))
    pred_s, pred_e = pred_s.data.numpy(), pred_e.data.numpy()
    pred_s, pred_e = pred_s.reshape((1, dims[0], dims[1])), pred_e.reshape((2, dims[0], dims[1]))
    x1, y1, x2, y2, move_half = gen_move_max(pred_s, pred_e, update['tile_grid'], update['player_index'])

    g.move(y1, x1, y2, x2, move_half=move_half)
    print("--- {} seconds --- in turn {}".format((time.time() - start_time), update['turn']))
