import math
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


GAMMA = 0.999
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 200
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STEPS = 0



Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward')
)


class HuberLoss(nn.Module):
    def __init__(self):
        super(HuberLoss, self).__init__()

    def forward(self, x, y, delta=0.5):
        err = t.abs(y - q)
        quad = err.clamp(0, 1)
        line = error - quad
        loss = t.mean(delta * quad**2 + line)

        return loss


def preprocess(frame):
    tsfm = [
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
    ]
    process = transforms.Compose(tsfm)
    img = process(frame)

    return img


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """save a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def select_action(model, state):
    global STEPS
    rand = random.random()
    # epsilon = get_epsilon_for_iteration(step)
    epsilon = get_epsilon(STEPS)

    # print(f"rand > epsilon: {rand>epsilon}")
    STEPS += 1
    if  rand > epsilon:
        with torch.no_grad():
            return model(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(4)]], device=DEVICE, dtype=torch.long)


def optimize_model(model, target, replay_memory, optimizer):
    """
        model: policy network
        target: target network
    """
    if len(replay_memory) < BATCH_SIZE:
        return
    transitions = replay_memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = model(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
    next_state_values[non_final_mask] = target(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)  # clip reward values, inplace
    optimizer.step()


def transform_reward(reward):
    r = t.tensor(reward)
    return r.clamp(-1, 1)


def get_epsilon(step):
    e = EPSILON_END + (EPSILON_START - EPSILON_END) * \
        math.exp(-1 * step / EPSILON_DECAY)

    return e


def get_screen(env):
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)

    return preprocess(screen).unsqueeze(0).to(DEVICE)



if __name__ == "__main__":
    mem = ReplayMemory(4)

    print(mem)
