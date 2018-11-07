import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_epsilon

sns.set_style('whitegrid')


EPSILON_START = 0.999
EPSILON_END = 0.01
EPSILON_DECAY = 200
TARGET_UPDATE = 10
EPISODES = 10000


def main():
    eps = [get_epsilon(e) for e in range(EPISODES)]
    plt.figure()
    plt.plot(eps)
    plt.xlim(0, EPISODES)
    plt.ylim(0, 1.0)
    plt.title('$\\varepsilon$-greedy Schedule')
    plt.xlabel('Episode')
    plt.ylabel('$\\varepsilon$')
    plt.savefig('assets/epsilon-schedule.png')
    print("Done")


if __name__ == "__main__":
    _ = main()
