import math
import random
import argparse
from itertools import count
from collections import namedtuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from src.models import AtariDQN
from src.utils import ReplayMemory, HuberLoss, Transition, save_checkpoint
from src.utils import select_action, optimize_model, get_screen, plot_durations


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('environment', type=str)
parser.add_argument('-episodes', '--N', type=int, default=10000)

args = parser.parse_args()

print(args)


ENVIRONMENT = args.environment

env = gym.make(ENVIRONMENT)  #.unwrapped
env.reset()

plt.ion()


BATCH_SIZE = 128
GAMMA = 0.999
EPSILON_START = 0.999
EPSILON_END = 0.01
EPSILON_DECAY = 200
TARGET_UPDATE = 10
EPISODES = args.N


def train():
    env.render()
    policy_net = AtariDQN(env.action_space.n).to(DEVICE)
    target_net = AtariDQN(env.action_space.n).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)
    episode_durations = []
    step = 0
    for episode in range(EPISODES):
        env.reset()
        last_screen = get_screen(env)
        current_screen = get_screen(env)
        state = current_screen - last_screen  # .to(DEVICE)
        for t in count():
            env.render()
            action = select_action(policy_net, state, step)
            step += 1
            a = action.item()
            frame, reward, is_done, info = env.step(a)
            reward = torch.tensor([reward], device=DEVICE)
            last_screen = current_screen
            current_screen = get_screen(env)
            if not is_done:
                next_state = current_screen - last_screen
            else:
                next_state = None
            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model(policy_net, target_net, memory, optimizer)
            if is_done:
                print(f"Step {step: 6d} | Episode {episode} | Action {a} | Reward {int(reward.item()):3d}")
                # Plotting
                episode_durations.append(t + 1)
                plot_durations(
                    f"{ENVIRONMENT}",
                    episode_durations,
                )
                break
        if episode % TARGET_UPDATE == 0 and episode != 0:
            save_checkpoint(
                f"{ENVIRONMENT}",
                {
                    'episode': episode + 1,
                    'step': step,
                    'state_dict': policy_net.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                },

            )
            target_net.load_state_dict(policy_net.state_dict())

    env.render()
    env.close()
    plt.ioff()
    plt.show()



if __name__ == "__main__":
    _ = train()
