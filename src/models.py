import torch as t
import torch.nn as nn
import torch.nn.functional as F


ATARI_SHAPE = (4, 105, 84)


class AtariDQN(nn.Module):
    def __init__(self):
        super(AtariDQN, self).__init__()

        self.n_actions = 4  # how to set this from gym environment?

        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(2592, 256)
        self.fc2 = nn.Linear(256, self.n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
