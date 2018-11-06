import random

import gym
import torch
import numpy as np

from src.utils import preprocess, transform_reward


def main():
    # Create a breakout environment
    env = gym.make('BreakoutDeterministic-v4')
    # Reset it, returns the starting frame
    frame = env.reset()
    # Render
    env.render()
    step = 0
    is_done = False
    while not is_done:
      # Perform a random action, returns the new frame, reward and whether the game is over
      frame, reward, is_done, info = env.step(env.action_space.sample())


      lives = info['ale.lives']


      print("step:", step)
      print("\tpreprocess", preprocess(frame))
      print("\ttransformed reward:", transform_reward(reward))

      step += 1

      # Render
      env.render()


if __name__ == "__main__":
    _ = main()
