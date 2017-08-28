import logging
import json
import threading
import time
import sys
from websocket import create_connection, WebSocketConnectionClosedException


EMPTY = -1
MOUNTAIN = -2
FOG = -3
OBSTACLE = -4

_ENDPOINT = "ws://botws.generals.io/socket.io/?EIO=3&transport=websocket"
_REPLAY_URL = "http://bot.generals.io/replays/"

_RESULTS = {
    "game_update": "",
    "game_won": "win",
    "game_lost": "lose",
}

user_id = "5900688366"
username = "[Bot] asdfshqwen123"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class Generals(object):

    # region is deprecated
    def __init__(self, userid, username, mode="1v1", gameid=None,
                 force_start=True, region=None):
        logging.debug("Creating connection")
        self._ws = create_connection(_ENDPOINT)
        self._lock = threading.RLock()

        logging.debug("Starting heartbeat thread")
        _spawn(self._start_sending_heartbeat)

        logging.debug("Joining game")
        self._send(["set_username", userid, username])

        if mode == "private":
            if gameid is None:
                raise ValueError("Gameid must be provided for private games")
            self._send(["join_private", gameid, userid])

        elif mode == "1v1":
            self._send(["join_1v1", userid])

        elif mode == "team":
            if gameid is None:
                raise ValueError("Gameid must be provided for team games")
            self._send(["join_team", gameid, userid])

        elif mode == "ffa":
            self._send(["play", userid])

        else:
            raise ValueError("Invalid mode")

        self._send(["set_force_start", gameid, force_start])

        self._seen_update = False
        self._move_id = 1
        self._start_data = {}
        self._stars = []
        self._map = []
        self._cities = []

    def move(self, y1, x1, y2, x2, move_half=False):
        if not self._seen_update:
            raise ValueError("Cannot move before first map seen")

        cols = self._map[0]
        a = y1 * cols + x1
        b = y2 * cols + x2
        self._send(["attack", a, b, move_half, self._move_id])
        print("move from {}, {} to {}, {}".format(x1, y1, x2, y2))
        self._move_id += 1

    def get_updates(self):
        while True:
            try:
                msg = self._ws.recv()
            except WebSocketConnectionClosedException:
                break

            if not msg.strip():
                break

            # ignore heartbeats and connection acks
            if msg in {"3", "40"}:
                continue

            # remove numeric prefix
            while msg and msg[0].isdigit():
                msg = msg[1:]

            msg = json.loads(msg)
            if not isinstance(msg, list):
                continue

            if msg[0] == "error_user_id":
                raise ValueError("Already in game")
            elif msg[0]== 'pre_game_start':
                logging.info("Game Prepare to Start")
            elif msg[0] == "game_start":
                logging.info("Game info: {}".format(msg[1]))
                self._start_data = msg[1]
            elif msg[0] == "game_update":
                yield self._make_update(msg[1])
            elif msg[0] in ["game_won", "game_lost"]:
                print(self._make_result(msg[0], msg[1]))
                break
            else:
                logging.info("Unknown message type: {}".format(msg))

    def close(self):
        self._ws.close()

    def _make_update(self, data):
        _apply_diff(self._map, data['map_diff'])
        _apply_diff(self._cities, data['cities_diff'])
        if 'stars' in data:
            self._stars[:] = data['stars']

        rows, cols = self._map[1], self._map[0]
        self._seen_update = True

        # sort by player index
        scores = {d['i']: d for d in data['scores']}
        scores = [scores[i] for i in range(len(scores))]
        return {
            'complete': False,
            'rows': rows,
            'cols': cols,
            'player_index': self._start_data['playerIndex'],
            'turn': data['turn'],
            'army_grid': [[self._map[2 + y*cols + x]
                          for x in range(cols)]
                          for y in range(rows)],
            'tile_grid': [[self._map[2 + cols*rows + y*cols + x]
                          for x in range(cols)]
                          for y in range(rows)],
            'lands': [s['tiles'] for s in scores],
            'armies': [s['total'] for s in scores],
            'alives': [not s['dead'] for s in scores],
            'generals': [(-1, -1) if g == -1 else (g // cols, g % cols)
                         for g in data['generals']],
            'cities': [(c // cols, c % cols) for c in self._cities],
            'usernames': self._start_data['usernames'],
            'teams': self._start_data.get('teams'),
            'stars': self._stars,
            'replay_url': _REPLAY_URL + self._start_data['replay_id'],
        }

    def _make_result(self, update, data):
        return {
            'complete': True,
            'result': update == "game_won",
            'player_index': self._start_data['playerIndex'],
            'usernames': self._start_data['usernames'],
            'teams': self._start_data.get('teams'),
            'stars': self._stars,
            'replay_url': _REPLAY_URL + self._start_data['replay_id'],
        }

    def _start_sending_heartbeat(self):
        while True:
            try:
                with self._lock:
                    self._ws.send("2")
            except WebSocketConnectionClosedException:
                break
            time.sleep(10)

    def _send(self, msg):
        try:
            with self._lock:
                self._ws.send("42" + json.dumps(msg))
        except WebSocketConnectionClosedException:
            pass


def _spawn(f):
    t = threading.Thread(target=f)
    t.daemon = True
    t.start()


def _apply_diff(cache, diff):
    i = 0
    a = 0
    while i < len(diff) - 1:

        # offset and length
        a += diff[i]
        n = diff[i+1]

        cache[a:a+n] = diff[i+2:i+2+n]
        a += n
        i += n + 2

    if i == len(diff) - 1:
        cache[:] = cache[:a+diff[i]]
        i += 1

    assert i == len(diff)
