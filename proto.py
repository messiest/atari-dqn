import itertools

import gym
from gym import wrappers

from models import AtariDQN

ENV = 'PhoenixDeterministic-v4'

env = gym.make(ENV)
env = wrappers.Monitor(env, 'playback/', force=True)

for i_episode in range(20):
    obs = env.reset()
    for t in itertools.count():
        env.render()
        a = env.action_space.sample()
        obs, r, is_done, info = env.step(a)
        if is_done:
            print(f"Episode finished after {t+1} steps")
            break

for video_path, meta_path in env.videos:
    print("VIDEOS:", video_path, meta_path)


print(AtariDQN == AtariDQN2)
