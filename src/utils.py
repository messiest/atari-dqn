import numpy as np
import torch
from torchvision import transforms


class HuberLoss(torch.nn.Module):
    def __init__(self):
        super(HuberLoss, self).__init__()

    def forward(self, x, y, delta=0.5):
        err = torch.abs(y - q)
        quad = err.clamp(0, 1)
        line = error - quad
        loss = torch.mean(delta * quad**2 + line)

        return loss


def preprocess(frame):
    img = torch.from_numpy(frame)
    transformations = [
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((110, 84)),
        transforms.ToTensor(),
    ]
    process = transforms.Compose(transformations)

    return process(img)


def transform_reward(reward):
    r = torch.tensor(reward)
    return r.clamp(-1, 1)
