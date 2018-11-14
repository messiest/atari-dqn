import random

import gym
import torch
import numpy as np
import torch.optim as optim

from utils import preprocess, transform_reward, ReplayMemory, HuberLoss
from models import AtariDQN
from atari import create_atari_env
from trainers import TargetNetwork


def main():
    pass


if __name__ == "__main__":
    env = create_atari_env('PongDeterministic-v4')
    # env = create_atari_env('CartPole-v0')
    model = AtariDQN
    memory = ReplayMemory(1000)
    trainer = TargetNetwork(env, model, optim.RMSprop, memory)

    for episode in range(100):
        trainer.train()

    trainer.close()
