import math
import random
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

from src.models import AtariDQN
from src.utils import ReplayMemory, HuberLoss, preprocess, select_action
from src.utils import Transition, optimize_model, get_screen, STEPS


env = gym.make('BreakoutDeterministic-v4').unwrapped
env.reset()


BATCH_SIZE = 128
GAMMA = 0.999
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 200
TARGET_UPDATE = 10
EPISODES = 10

STEP = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    env.render()

    policy_net = AtariDQN()
    target_net = AtariDQN()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)

    for i in range(EPISODES):
        env.reset()
        last_screen = get_screen(env)
        current_screen = get_screen(env)

        state = current_screen - last_screen

        for t in count():
            action = select_action(policy_net, state)
            STEP += 1
            a = action.item()

            frame, reward, is_done, info = env.step(a)
            env.render()
            reward = torch.tensor([reward])

            last_screen = current_screen
            current_screen = get_screen(env)

            if not is_done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            print(f"Step {STEP} | Episode {i} | Action {a} | Reward {reward.item():3d}")

            memory.push(state, action, next_state, reward)

            state = next_state

            optimize_model(policy_net, target_net, memory, optimizer)
            if is_done:
                # Plotting
                # episode_durations.append(t + 1)
                # plot_durations()
                break
        if i % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    env.render()
    env.close()
