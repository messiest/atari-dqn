from collections import deque

import numpy as np
import gym
from gym import spaces
from PIL import Image
import cv2
import torch
from torchvision import transforms


class ProcessFrameAtari(gym.Wrapper):
    def __init__(self, env=None):
        super(ProcessFrameAtari, self).__init__(env)

    def step(self, action):
        observation, reward, is_done, info = self.env.step(action)

        return self.preprocess(observation), reward, is_done, info

    def reset(self):
        return self.preprocess(self.env.reset())

    def preprocess(self, frame):
        tsfm = [
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
        ]
        process = transforms.Compose(tsfm)
        img = process(frame)

        return img

    def get_screen(self):
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        return self.preprocess(screen).unsqueeze(0)


def wrap_atari(env):
    env = ProcessFrameAtari(env)
    return env


def create_atari_env(env_id):
    env = gym.make(env_id)
    env = wrap_atari(env)
    return env
