from generalsim import GeneralBase
import torch
from torch.autograd import Variable
import CNNLSTMPolicy
import numpy as np

##### Environment settings
MAP_MIN = 17
MAP_MAX = 23
MOUNTAIN_RATIO = 0.2
CITY_NUM = 9
CITY_MIN = 40
CITY_MAX = 50
MOUNTAIN = -2
CITY = -1


class GeneralEnvironment(GeneralBase):
    """Class for simulating generals.io game against a policy bot
       Currently only 1 v 1 is supported"""
    def __init__(self, model_path):
        super(GeneralEnvironment, self).__init__()

        model = CNNLSTMPolicy.CNNLSTMPolicy()
        model.load_state_dict(torch.load(model_path))
        model = model.eval()
        self.model = model

        self.init_board()

    def gen_move_max(self, pred_start, pred_end, index):
        label_map = self.label_map
        army_map = self.army_map
        max_prob = -1 * float('inf')
        row = pred_start.shape[1]
        col = pred_start.shape[2]

        x1, y1, x2, y2, move_half = 0, 0, 0, 0, False

        for y in range(row):
            for x in range(col):
                if label_map[y, x] != index + 1 or army_map[y, x] < 2:
                    continue

                start_prob = pred_start[0, y, x]
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    x_new = x + dx
                    y_new = y + dy

                    if x_new < 0 or x_new >= col or y_new < 0 or y_new >= row:
                        continue

                    if label_map[y_new, x_new] == MOUNTAIN:
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


    def init_board(self):
        """Initializes a random baord"""
        self.map_height = np.random.randint(MAP_MIN, MAP_MAX)
        self.map_width = np.random.randint(MAP_MIN, MAP_MAX)

        self.label_map = np.zeros((self.map_height,
                                   self.map_width)).astype(int)
        self.army_map = np.zeros((self.map_height,
                                  self.map_width)).astype(int)

        tile_num = self.map_height * self.map_width
        perm = np.random.permutation(tile_num)

        mountain_num = int(MOUNTAIN_RATIO * tile_num)
        city_bound = mountain_num + CITY_NUM
        city_val = np.random.randint(CITY_MIN, CITY_MAX, size=CITY_NUM)
        self.mountains = perm[:mountain_num]
        self.cities = perm[mountain_num: city_bound]
        self.generals = perm[city_bound: city_bound + 2]

        # label_map represents state of the board.
        # -2 represents mountains
        # -1 represents neutral cities
        # 0 will be used to indicate unoccupied tiles
        # 1 - num_players will indicate possession by respective player

        self.label_map.flat[self.mountains] = MOUNTAIN
        self.label_map.flat[self.cities] = CITY
        self.army_map.flat[self.cities] = city_val

        self.label_map.flat[self.generals[0]] = 1
        self.label_map.flat[self.generals[1]] = 2
        self.army_map.flat[self.generals] += 1

        self.model.init_hidden(self.map_height, self.map_width)

        self.turn_num = 0
        self.player_land_num = 0
        self.player_army_num = 0

        # Keep a map of the index generals to there original start locations
        self.gen_index_to_coord = {i: coord for i,
                                   coord in enumerate(self.generals)}
        self.taken_cities = np.array([])

    def model_move(self):
        state = self.export_state(1)
        state = state[np.newaxis, ...]
        pred_start, pred_end = self.model.forward(Variable(torch.Tensor(state)))
        pred_start, pred_end = pred_start.data.numpy(), pred_end.data.numpy()
        pred_start = pred_start.reshape((1, self.map_height, self.map_width))
        pred_end = pred_end.reshape((2, self.map_height, self.map_width))

        x1, y1, x2, y2, move_half = self.gen_move_max(pred_start, pred_end, 1)

        start, end = x1 + y1 * self.map_width, x2 + y2 * self.map_width
        move = {"start": start, "end": end, "is50": move_half}
        return move

    def step(self, action):
        """
        Roughly follows the API of OpenAI gym

        Keyword Arguments:
            a flat index of 8 x w x h array indicating
            movement direction
        Returns:
            observation, reward, done, info
        """
        move = self._parse_action(action)
        self.turn_num += 1
        reward = self.move(move, player_index=0)
        self.move(self.model_move(), player_index=1)
        self.increment_count()

        army_num, land_num = self.compute_stats(0)
        reward += (army_num - self.player_army_num) + (land_num - self.player_land_num)
        # print("This is the land difference: {}".format(land_num - self.player_land_num))
        done = (army_num == 0)
        state = self.export_state(0)

        self.player_army_num = army_num
        self.player_land_num = land_num

        return state, reward, done, {}

    def reset(self):
        self.init_board()
        return self.export_state(0)

    def _parse_action(self, action):
        move_type, y, x = np.unravel_index(action, (8, self.map_height, self.map_width))
        start = y * self.map_width + x
        index = move_type % 4

        if index == 0:
            end = start + self.map_width
        elif index == 1:
            end = start + 1
        elif index == 2:
            end = start - self.map_width
        elif index == 3:
            end = start - 1
        else:
            raise("invalid index")

        is_50 = True if index >= 4 else False

        return {'start': start, 'end': end, 'is50': is_50}



if __name__ == "__main__":
    general_env = GeneralEnvironment("2_epoch.mdl")
    for i in range(20):
        _, reward, _, _ = general_env.step({'start': 0, 'end': 1, 'is50': False})
        print(reward)
    general_env.reset()
    for i in range(20):
        _, reward, _, _ = general_env.step({'start': 0, 'end': 1, 'is50': False})
        print(reward)
    print(general_env)
