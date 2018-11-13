import gym
import torch
from torch.utils.data import Dataset, DataLoader


class GymDataset(Dataset):
    """
        An attempt to create an environment agnostic Dataset for OpenAI's gym
    """

    def __init__(self, environment_name, capacity):
        self.env = gym.make(environment_name).unwrapped
        self.observations = []
        self.position = 0

        self.step = 0

        self.screen = None
        self.state = None

    def __str__(self):
        return f"{self.env}"

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        self.step += 1
        print(self.step)
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))

        return screen

    def push(self, *args):
        """save a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def start(self):
        self.env.reset()

    def render(self):
        return self.env.render()

    def sample(self, batch_size):
        return random.sample(self.observations, batch_size)

    def random(self):
        return self.env.action_space.sample()


TEST_ENVIRONMENT = 'CartPole-v1'


if __name__ == "__main__":
    gym_data = GymDataset(TEST_ENVIRONMENT, 10000)
    print(gym_data.start())
    print(gym_data)

    for i in gym_data:
        print(i, gym_data.random())
