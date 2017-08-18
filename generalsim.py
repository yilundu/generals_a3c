import json
import numpy as np
import math


class GeneralSim(object):
    def __init__(self, path):
        """Initializes generals game from a gioreplay file"""
        replay = json.load(open(path, "rb"))
        self.map_width = replay['mapWidth']
        self.map_height = replay.get('mapHeight', self.map_width)
        self.stars = replay['stars']
        self.num_players = len(self.stars)
        self.moves = replay['moves']
        self.version = replay['version']
        # moves_index represents index of next move that will happen
        self.moves_index = 0
        self.replay = replay
        self.afks = replay['afks']
        self.afks_index = 0
        self.afks_count = {}

        # Initializes starting tiles of players
        self.init_board()

    def init_board(self):
        self.turn_num = 0
        self.generals = np.array(self.replay['generals'])
        self.cities = np.array(self.replay['cities'])
        self.taken_cities = np.array([])
        self.mountains = np.array(self.replay['mountains'])
        # label_map represents state of the board.
        # -2 represents mountains
        # -1 represents neutral cities
        # 0 will be used to indicate unoccupied tiles
        # 1 - num_players will indicate possession by respective player
        self.label_map = np.zeros((self.map_height,
                                   self.map_width)).astype('int')
        self.label_map.flat[self.generals] = np.arange(
            1, len(self.generals) + 1)
        self.label_map.flat[self.mountains] = -2
        self.label_map.flat[self.cities] = -1

        # Represents the army count at each different tile
        self.army_map = np.zeros((self.map_height,
                                  self.map_width)).astype('int')
        self.army_map.flat[self.generals] += 1
        self.army_map.flat[self.cities] = self.replay['cityArmies']

        # Keep a map of the index generals to there original start locations
        self.gen_index_to_coord = {i: coord for i,
                                   coord in enumerate(self.generals)}

    def step(self):
        self.turn_num += 1
        self.move_steps()
        self.increment_count()
        self.afk_remove()

    def afk_remove(self):
        while self.afks_index < len(self.afks) and \
                self.afks[self.afks_index]['turn'] < self.turn_num:
            index = self.afks[self.afks_index]['index']
            self.afks_count[index] = self.afks_count.get(index, 0) + 1
            self.afks_index += 1

            if self.afks_count[index] == 2:
                self.generals = self.generals[
                    self.generals != self.gen_index_to_coord[index]]
                self.label_map[self.label_map == index + 1] = -1

    def move_steps(self):
        while self.moves_index < len(self.moves) and \
                self.moves[self.moves_index]['turn'] <= self.turn_num:

            move = self.moves[self.moves_index]
            self.moves_index += 1

            start = move['start']
            end = move['end']

            reserve = math.ceil(self.army_map.flat[start] / 2.) \
                if move['is50'] else 1
            attack_force = self.army_map.flat[start] - reserve
            self.army_map.flat[start] = reserve

            start_label = self.label_map.flat[start]
            end_label = self.label_map.flat[end]
            end_army_value = self.army_map.flat[end]

            if start_label == end_label:
                self.army_map.flat[end] += attack_force
            else:
                if end_army_value >= attack_force:
                    self.army_map.flat[end] -= attack_force
                else:
                    self.label_map.flat[end] = start_label
                    self.army_map.flat[end] = attack_force - end_army_value
                    if end in self.cities:
                        self.cities = self.cities[self.cities != end]
                        self.taken_cities = np.insert(self.taken_cities,
                                                      len(self.taken_cities),
                                                      end).astype('int')
                    elif end in self.generals:
                        defeated_land = (self.label_map == end_label)
                        self.label_map[defeated_land] = start_label
                        self.army_map[defeated_land] = (
                            self.army_map[defeated_land] + 1) / 2
                        self.generals = self.generals[
                            self.generals != self.gen_index_to_coord[
                                end_label - 1]]
                        self.taken_cities = np.insert(self.taken_cities,
                                                      len(self.taken_cities),
                                                      end).astype('int')

    def export_state(self, index):
        """Given the index of specific user, exports the view of the board,
        turn_number % 50 and the the army number of all other players """
        # TODO implement this for multiplayers
        # TODO: normalize features

        """Current Schema for game 1v1 game state:
                channel 0: army values of user
                channel 1: observed army values of opponent
                channel 2: binary values indicating obstacle
                channel 3: army values of observed neutral cities
                channel 4: coordinates of observed mountains
                channel 5: coordinates with values of capital
                channel 6: 1s where location is unobserved
                channel 7: coordinates of cities owned by self
                channel 8: coordinates owned by enemy player
                channel 9: turn number % 50
                channel 10: enemy soldier number relative to own"""

        export_state = np.zeros((11, self.map_height, self.map_width))
        label = index + 1

        # We can only one unit away in generals so mask all tiles outside
        view_mask = (self.label_map == label)
        bool_mask = view_mask.copy()

        view_mask[:, :-1] += bool_mask[:, 1:]
        view_mask[:, 1:] += bool_mask[:, :-1]
        view_mask[:-1, :] += bool_mask[1:, :]
        view_mask[1:, :] += bool_mask[:-1, :]
        view_mask[:-1, :-1] += bool_mask[1:, 1:]
        view_mask[1:, 1:] += bool_mask[:-1, :-1]
        view_mask[:-1, 1:] += bool_mask[1:, :-1]
        view_mask[1:, :-1] += bool_mask[:-1, 1:]

        label_mask = (self.label_map == label)
        enemy_global_mask = ((self.label_map > 0) * ~label_mask)
        export_state[0, label_mask] = self.army_map[label_mask]

        index_label_map = np.zeros((self.map_height, self.map_width))
        index_label_map[view_mask] = self.label_map[view_mask]

        enemy_mask = (index_label_map > 0) * (index_label_map != label)
        export_state[1, enemy_mask] = self.army_map[enemy_mask]

        blockade_map = (self.label_map < 0)
        export_state[2, blockade_map] = 1

        observed_n_city_mask = (self.label_map == -1) * view_mask
        export_state[3,
                     observed_n_city_mask] = self.army_map[
                         observed_n_city_mask]

        observed_mountain_mask = (self.label_map == -2) * view_mask
        export_state[4, observed_mountain_mask] = 1

        ob_gen_mask = self.generals[view_mask.flat[self.generals]]
        export_state[5].flat[ob_gen_mask] = self.army_map.flat[ob_gen_mask]

        export_state[6] = (~view_mask).astype(int)

        if len(self.taken_cities) > 0:
            taken_city_mask = self.taken_cities[
                view_mask.flat[self.taken_cities]]
            if len(taken_city_mask) > 0:
                fri_city_mask = taken_city_mask[
                    label_mask.flat[taken_city_mask]]
                enem_city_mask = taken_city_mask[
                    enemy_global_mask.flat[taken_city_mask]]
                if len(fri_city_mask) > 0:
                    export_state[7].flat[fri_city_mask] = self.army_map.flat[
                        fri_city_mask]
                if len(enem_city_mask) > 0:
                    export_state[8].flat[enem_city_mask] = self.army_map.flat[
                        enem_city_mask]

        export_state[9] = (self.turn_num % 50)

        num_troops = self.army_map[label_mask].sum()
        enem_num_troops = self.army_map[enemy_global_mask].sum()
        export_state[10] = num_troops / 1. / enem_num_troops

    def increment_count(self):
        # Every two turns each city increases the number units in capital
        if self.turn_num % 2 == 1:
            self.army_map.flat[self.generals] += 1
            if self.taken_cities.shape[0] > 0:
                self.army_map.flat[self.taken_cities] += 1
            if self.version == 5:
                regen_cities = self.cities[
                    self.army_map.flat[self.cities] < 40]
                self.army_map.flat[regen_cities] += 1

        # Every fifty turns every tile gains a unit
        if self.turn_num % 50 == 49:
            self.army_map[self.label_map > 0] += 1

    def __str__(self):
        output_text = ""
        output_text += "Printing the label_map... \n"
        output_text += str(self.label_map) + "\n"
        output_text += "Printing the army_map... \n"
        output_text += str(self.army_map) + "\n"
        output_text += "Playing turn number {}".format(self.turn_num)
        return output_text


if __name__ == "__main__":
    example_game = GeneralSim("rFUuZ8evl.gioreplay")
    for _ in range(400):
        example_game.step()

    example_game.export_state(5)

    print(example_game)
