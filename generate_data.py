import argparse
from threading import Thread, Lock
from Queue import Queue
from termcolor import cprint
from generalsim import GeneralSim

from os import listdir
from os.path import isfile, join

BATCH_INTERVAL = 100


def extract_data(l_f, threads, thresh, players):
    """Extracts data from a list of  gioreplay"""
    lock = Lock()
    x, y = [], []

    q = Queue()

    def extract_games():
        x_temp, y_temp = [], []
        counter = 0
        while True:
            f_name = q.get()
            if f_name.endswith(".gioreplay"):
                game = GeneralSim(f_name)
                status = game.add_log(thresh, players)

                if status:
                    while not game.step():
                        continue

                    game_x, game_y = game.export_log()

                    x_temp.extend(game_x)
                    y_temp.extend(game_y)
                    counter += 1

                    if counter % BATCH_INTERVAL == 0:
                        cprint("Added {} files!".format(BATCH_INTERVAL), 'green')
                        counter = 0

                        lock.acquire()
                        x.extend(x_temp)
                        y.extend(y_temp)
                        lock.release()

                        x_temp = []
                        y_temp = []

            q.task_done()

    for i in range(threads):
        t = Thread(target=extract_games)
        t.daemon = True
        t.start()

    for f_name in iter(l_f):
        q.put(f_name)

    q.join()

    return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--data", type=str, default="replays_prod",
                        help="Directory where the gioreplay files are stored")
    parser.add_argument("--stars", type=int, default=10)
    parser.add_argument("--players", type=int, default=2)
    args = parser.parse_args()

    cprint("Finding all gioreplay files...", "green")
    f_list = [join(args.data, f) for f in listdir(args.data) if isfile(join(args.data, f))]

    cprint("Extracting data from all gioreplay files...", "green")
    x, y = extract_data(f_list, args.threads, args.stars, args.players)
