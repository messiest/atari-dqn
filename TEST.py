import random

import gym
import torch
import numpy as np

from src.utils import preprocess, transform_reward, ReplayMemory, HuberLoss
from src.models import AtariDQN


def main():
    # Create a breakout environment
    env = gym.make('BreakoutDeterministic-v4')
    # Reset it, returns the starting frame
    frame = env.reset()
    # Render
    env.render()
    step = 0
    is_done = False
    mem = ReplayMemory(4)
    while not is_done:
        # Perform a random action, returns the new frame, reward and whether the game is over
        # a_space = env.action_space
        # print("a_space", a_space.shape)

        a = env.action_space.sample()
        print("\taction", a)

        frame, reward, is_done, info = env.step(a)

        mem.push(frame, a, frame, reward)

        lives = info['ale.lives']

        print("\tpreprocess", preprocess(frame))
        print("\ttransformed reward:", transform_reward(reward))

        if is_done:
            break

        step += 1
        # Render
        env.render()

    print(mem)


if __name__ == "__main__":
    _ = main()
