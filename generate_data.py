import argparse
from multiprocessing import Pool
from termcolor import cprint
from generalsim import GeneralSim

from os import listdir
from os.path import isfile, join

REPORT_INTERVAL = 10000
# Currently only extracting games between 2 players
# with greater than 80 stars
NUM_PLAYERS = 2
STAR_TRESH = 80


def extract_game(f_name):
    game_x, game_y = [], []
    if f_name.endswith(".gioreplay"):
        game = GeneralSim(f_name)
        status = game.add_log(STAR_TRESH, NUM_PLAYERS)

        if status:
            while not game.step():
                pass

            game_x, game_y = game.export_log()

    return True


def extract_data(l_f, threads):
    """Extracts data from a list of  gioreplay"""
    pool = Pool(threads)

    mapped_data = pool.map(extract_game, l_f)
    x, y = zip(*mapped_data)

    return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--data", type=str, default="replays_prod",
                        help="Directory where the gioreplay files are stored")
    parser.add_argument("--stars", type=int, default=10)
    parser.add_argument("--players", type=int, default=2)
    args = parser.parse_args()
    NUM_PLAYERS = args.players
    STAR_TRESH = args.stars

    cprint("Finding all gioreplay files...", "green")
    f_list = [join(args.data, f) for f in listdir(args.data) if isfile(join(args.data, f))]

    cprint("Extracting data from all gioreplay files...", "green")
    x, y = extract_data(f_list, args.threads)


