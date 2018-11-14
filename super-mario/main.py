import os
import random
import argparse
import multiprocessing

import numpy as np
import tqdm
import gym
import torch
import torch.multiprocessing as _mp

from actor_critic import ActorCritic
from shared_adam import SharedAdam
from mario_wrapper import create_mario_env
from mario_actions import ACTIONS
from a3c import train, test
from utils import FontColor


parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards (default: 0.9)')
parser.add_argument('--tau', type=float, default=1.00, help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=250, help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=4, help='random seed (default: 4)')
parser.add_argument('--num-processes', type=int, default=multiprocessing.cpu_count(), help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=50, help='number of forward steps in A3C (default: 50)')
parser.add_argument('--max-episode-length', type=int, default=1000000, help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='SuperMarioBros-1-1-v3', help='environment to train on (default: SuperMarioBros-1-1-v3)')
parser.add_argument('--no-shared', default=False, help='use an optimizer without shared momentum.')
parser.add_argument('--use-cuda', default=True, help='run on gpu.')
parser.add_argument('--record', default=False, help='record playback of tests')
parser.add_argument('--save-interval', type=int, default=100, help='model save interval (default: 10)')
parser.add_argument('--non-sample', type=int,default=2, help='number of non sampling processes (default: 2)')

args = parser.parse_args()

mp = _mp.get_context('spawn')

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    env = create_mario_env(args.env_name)
    # env = gym.wrappers.Monitor(env, "playback", force=True)
    shared_model = ActorCritic(env.observation_space.shape[0], len(ACTIONS))

    if torch.cuda.is_available():
        shared_model.cuda()

    shared_model.share_memory()

    torch.manual_seed(args.seed)

    optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)

    print(FontColor.BLUE + f"Number of available cores: {mp.cpu_count(): 2d}" + FontColor.END)

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    # pbar = tqdm.tqdm(total=100)
    # def pbar_update(*a):
    #     pbar.update()

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter))

    p.start()
    processes.append(p)

    num_processes = args.num_processes
    no_sample = args.non_sample  # count of non-sampling processes

    if args.num_processes > 1:
        num_processes = args.num_processes - 1

    sample_val = num_processes - no_sample

    for rank in range(0, num_processes):
        if rank < sample_val:  # random action
            p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
        else:  # best action
            p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer, False))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
