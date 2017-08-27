from websocket import create_connection
import logging
import sys
import json
import threading
import time
import numpy as np

ENDPOINT = "ws://botws.generals.io/socket.io/?EIO=3&transport=websocket"
user_id = "5900688366"
username = "[Bot] asdfshqwen123"

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class GeneralsClient(object):

    def __init__(self, user_id, username, gameid=None,
                 force_start=True):
        self._ws = create_connection(ENDPOINT)

        logging.debug("Starting update thread")
        _spawn_thread(self.get_updates)

        logging.debug("Starting heartbeat thread.")
        _spawn_thread(self.init_heartbeat_thread)

        logging.debug("Setting username...")
        self._send(["set_username", user_id, username])

        logging.debug('Trying to join game...')
        self._send(["join_private", gameid, user_id])

        self._send(["set_force_start", gameid, force_start])

    def _send(self, payload):
        self._ws.send("42" + json.dumps(payload))

    def get_updates(self):
        while True:
            msg = self._ws.recv()

            if not msg.strip():
                break

            if msg in {"3", "40"}:
                logging.debug("Received heartbeat or connection ack")

            while msg and msg[0].isdigit():
                msg = msg[1:]

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
                yield self._make_result(msg[0], msg[1])
                break
            else:
                logging.info("Unknown message type: {}".format(msg))

        logging.debug("Exiting the get_updates loop")

    def init_heartbeat_thread(self):
        while True:
            self._send(2)
            time.sleep(10)


def _spawn_thread(fn):
    t = threading.Thread(target=fn)
    t.daemon = True
    t.start()


if __name__ == "__main__":
    client = GeneralsClient(user_id, username, gameid='avko')
    time.sleep(360)
