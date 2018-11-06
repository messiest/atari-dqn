import numpy as np
import torch as t
from torch import nn
from torchvision import transforms


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
    img = t.from_numpy(frame)
    transformations = [
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((110, 84)),
        transforms.ToTensor(),
    ]
    process = transforms.Compose(transformations)

    return process(img)


def transform_reward(reward):
    r = t.tensor(reward)
    return r.clamp(-1, 1)
