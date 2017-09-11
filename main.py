import argparse
import os

import torch
import torch.multiprocessing as mp

import my_optim
from generalsenv import GeneralEnvironment
from ActorCritic import ActorCritic
from test import test
from a3c_trainer import train

# Based on
# https://github.com/ikostrikov/pytorch-a3c
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='learning rate (default: 0.00001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=25,
                    help='value loss coefficient (default: 25)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=16,
                    help='how many training processes to use (default: 16)')
parser.add_argument('--num-steps', type=int, default=30,
                    help='number of forward steps in A3C (default: 30)')
parser.add_argument('--max-episode-length', type=int, default=500,
                    help='maximum length of an episode (default: 500')
parser.add_argument('--no-shared', default=True,
                    help='use an optimizer without shared momentum.')
parser.add_argument('--off-tile-coef', type=float, default=10,
                   help='weight to penalize bad movement')
parser.add_argument('--checkpoint-interval', type=float, default=None,
                   help='interval to save model')


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'

    args = parser.parse_args()

    env = GeneralEnvironment('2_epoch.mdl')
    shared_model = ActorCritic()

    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    processes = []

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
