import json
import numpy as np
import math
import traceback


class GeneralBase(object):

    def __init__(self):
        self.log_players = []
        self.taken_cities = np.array([])
        self.version = 7

    def add_log(self, thresh_stars, players, **kwargs):
        """Specifies minimum condition for which we log data for players"""
        if self.stars:
            if len(self.stars) == players:
                for i, stars in enumerate(self.stars):
                    if stars >= thresh_stars:
                        self.log_players[i] = True
                        # We store the data of each player in two different dictionaries
                        self.player_datasets[i] = ([], [], [])

        status = True if self.log_players else False
        return status

    def move(self, move, player_index=None):
        start = move['start']
        end = move['end']
        reward = 0.0

        if not self.is_valid_move(start, end, player_index):
            return (reward - 1)

        start_label = self.label_map.flat[start]
        end_label = self.label_map.flat[end]

        # mountains are represented by -2
        if end_label == -2:
            return (reward - 1)

        index = start_label - 1

        if (index) in self.log_players:
            state = self.export_state(index)
            # The output of our model with be a plane of 3 convolutional
            # outputs, where the first indicates the originating unit and
            # the other indicating target destination and whether it is a
            # half or full output respectively.
            train_end = end
            if move['is50']:
                train_end += len(self.label_map.flat)

            self.player_datasets[index][0].append(state.astype(np.float16))
            self.player_datasets[index][1].append(start)
            self.player_datasets[index][2].append(train_end)

        reserve = math.ceil(self.army_map.flat[start] / 2.) \
            if move['is50'] else 1
        attack_force = self.army_map.flat[start] - reserve
        self.army_map.flat[start] = reserve

        end_army_value = self.army_map.flat[end]

        if start_label == end_label:
            self.army_map.flat[end] += attack_force
        else:
            if end_army_value >= attack_force:
                self.army_map.flat[end] -= attack_force
            else:
                self.label_map.flat[end] = start_label
                self.army_map.flat[end] = attack_force - end_army_value

                reward += 1.0
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
                    reward += 20.0

                if end in self.taken_cities:
                    reward += 4.0
        return reward

    def is_valid_move(self, start, end, player_index):
        start_label = self.label_map.flat[start]

        if end < len(self.label_map.flat) and end >= 0:
            end_label = self.label_map.flat[end]
        else:
            return False

        index = start_label - 1

        if player_index != None and (player_index != index):
            return False

        if self.army_map.flat[start] == 0:
            return False

        start_x, start_y = np.unravel_index(start, (self.map_height, self.map_width))
        end_x, end_y = np.unravel_index(end, (self.map_height, self.map_width))

        if abs(start_x - end_x) + abs(start_y - end_y) != 1:
            return False

        return True


    def compute_stats(self, index):
        """Returns the army_num, land_num of index respectively"""
        index_map = (self.label_map == index + 1)
        army_num = self.army_map[index_map].sum()
        land_num = index_map.sum()
        return army_num, land_num

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
                channel 8: coordinates of cities owned by enemy player
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

        export_state[9] = (self.turn_num % 50) / 50.0

        num_troops = self.army_map[label_mask].sum()
        enem_num_troops = self.army_map[enemy_global_mask].sum()
        export_state[10] = min(num_troops / 1. / max(enem_num_troops, 1), 10.) / 10.

        return export_state

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

    def export_log(self):
        x, y, z = [], [], []
        for index, value in self.player_datasets.iteritems():
            x.append(np.array(value[0]))
            y.append(np.array(value[1]))
            z.append(np.array(value[2]))
        return x, y, z

    def __str__(self):
        output_text = ""
        output_text += "Printing the label_map... \n"
        output_text += str(self.label_map) + "\n"
        output_text += "Printing the army_map... \n"
        output_text += str(self.army_map) + "\n"
        output_text += "Playing turn number {}".format(self.turn_num)
        return output_text


class GeneralSim(GeneralBase):
    """Class for simulating games from gioreplay files"""
    def __init__(self, path):
        """Initializes generals game from a gioreplay file"""
        replay = json.load(open(path, "rb"))
        self.map_width = replay['mapWidth']
        self.map_height = replay.get('mapHeight', self.map_width)
        self.stars = replay['stars']

        if self.stars:
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

        # Internal state to control what players we should log
        self.log_players = {}
        self.player_datasets = {}

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
        try:
            self.turn_num += 1
            is_end = self.move_steps()
            self.increment_count()
            self.afk_remove()
        except Exception:
            traceback.print_exc()
            raise

        return is_end

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
            self.move(move)
            self.moves_index += 1

        is_end = False if self.moves_index < len(self.moves) and self.turn_num < 800 else True
        return is_end

if __name__ == "__main__":
    # Example to simulate a generals game
    example_game = GeneralSim("rFUuZ8evl.gioreplay")
    example_game.add_log(10, 7)
    for _ in range(400):
        example_game.step()
    print(example_game)
