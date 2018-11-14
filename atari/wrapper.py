from collections import deque

import numpy as np
import gym
from gym import spaces
from PIL import Image
import cv2
import torch
from torchvision import transforms


def _preprocess(frame):
    tsfm = [
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
    ]
    process = transforms.Compose(tsfm)
    img = process(frame)

    return img


class ProcessFrameAtari(gym.Wrapper):
    def __init__(self, env=None):
        super(ProcessFrameAtari, self).__init__(env)

    def step(self, action):
        observation, reward, is_done, info = self.env.step(action)

        return _preprocess(observation), reward, is_done, info

    def reset(self):
        return _preprocess(self.env.reset())


class FrameBuffer(gym.Wrapper):
    def __init__(self, env=None, skip=4, shape=(84, 84)):
        super(FrameBuffer, self).__init__(env)
        self.counter = 0
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
        self.skip = skip
        self.buffer = deque(maxlen=self.skip)

    def step(self, action):
        obs, reward, is_done, info = self.env.step(action)
        counter = 1
        total_reward = reward
        self.buffer.append(obs)

        for i in range(self.skip-1):
            if not is_done:
                obs, reward, is_done, info = self.env.step(action)
                total_reward += reward
                counter += 1
                self.buffer.append(obs)
            else:
                self.buffer.append(obs)

        frame = np.stack(self.buffer, axis=0)
        frame = np.reshape(frame, (4, 84, 84))

        return frame, total_reward, is_done, info

    def reset(self):
        self.buffer.clear()
        obs = self.env.reset()
        for i in range(self.skip):
            self.buffer.append(obs)

        frame = np.stack(self.buffer, axis=0)
        frame = np.reshape(frame, (4, 84, 84))

        return frame


class Utilities(gym.Wrapper):
    def __init__(self, env=None):
        super(Utilities, self).__init__(env)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def get_screen(self):
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        return _preprocess(screen).unsqueeze(0)


def wrap_atari(env):
    env = ProcessFrameAtari(env)
    env = FrameBuffer(env)
    env = Utilities(env)
    return env


def create_atari_env(env_id):
    env = gym.make(env_id)
    env = wrap_atari(env)
    return env
