import os
import math
import shutil
import random
import fnmatch
from collections import deque, namedtuple

import numpy as np
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from utils import get_epsilon

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    print("iPython")
    from IPython import display
plt.ion()

sns.set_style('whitegrid')


def plot_durations(env_name, durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy(), alpha=0.5)
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

    if len(durations_t) % 10 == 0 and len(durations_t) != 0:
        plt.savefig(f'assets/{env_name}_durations.png')


def plot_rewards(env_name, rewards):
    plt.figure(2)
    plt.clf()
    rewards_t = torch.tensor(rewards, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy(), alpha=0.5)
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

    if len(rewards_t) % 10 == 0 and len(rewards_t) != 0:
        plt.savefig(f'assets/{env_name}_rewards.png')


def plot_epsilon_schedule(episodes):
    eps = [get_epsilon(e) for e in range(episodes)]
    plt.figure()
    plt.plot(eps)
    plt.xlim(0, EPISODES)
    plt.ylim(0, 1.0)
    plt.title('$\\varepsilon$-greedy Schedule')
    plt.xlabel('Episode')
    plt.ylabel('$\\varepsilon$')
    plt.savefig('assets/epsilon-schedule.png')
