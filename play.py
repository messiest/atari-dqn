import os

from models import AtariDQN
from utils import load_checkpoint


CHECKPOINT_PATH = os.path.join("checkpoints", "CartPole-v1_71.pth.tar")


if __name__ == "__main__":
    model_state, optim_state, episode = load_checkpoint(CHECKPOINT_PATH)
    print(model_state)
    print(optim_state)
    print(episode)

    # model = AtariDQN()
