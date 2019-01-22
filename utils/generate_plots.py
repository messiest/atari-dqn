import os
import sys
import argparse
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

from parsers import parse_loss_logs, parse_result_logs


parser = argparse.ArgumentParser('Mario.ai Plotting')
parser.add_argument('--env-name', type=str, default='SuperMarioBrosNoFrameskip-v0', help='environment name to generate plots for')
parser.add_argument('--model-id', type=str, default='murder_log')
parser.add_argument('--log-dir', type=str, default='logs/')
args = parser.parse_args()


def plot_episode_results(args, window=100):
    print(f"Plotting {args.model_id}'s results...")

    log_dir = os.path.join(args.log_dir, args.env_name)
    assert os.path.exists(log_dir), 'File not found'

    data = parse_result_logs(args.model_id, args.env_name, args.log_dir)

    for session in data:
        # saving plots
        save_dir = os.path.join(log_dir, args.model_id, session, 'plots')
        os.makedirs(save_dir, exist_ok=True)

        df = pd.DataFrame().from_dict(data[session])
        plt.figure(figsize=(20, 12), dpi=256)

        plt.plot(
            df.index,
            df['reward'].rolling(window).mean(),
            label=f"Session ID: {session}",
        )

        plt.suptitle(args.env_name, fontsize=18, y=.925)
        plt.ylabel(f'Reward \n ({window}-Episode Rolling Average)')
        plt.xlabel('Episode')
        plt.legend()

        plt.savefig(os.path.join(save_dir, 'results.png'))


def plot_episode_loss(args, window=1000):
    print(f"Plotting {args.model_id}'s loss...")

    log_dir = os.path.join(args.log_dir, args.env_name)
    assert os.path.exists(log_dir), 'File not found'

    data = parse_loss_logs(args.model_id, args.env_name, args.log_dir)

    for session in data:
        # saving plots
        save_dir = os.path.join(log_dir, args.model_id, session, 'plots')
        os.makedirs(save_dir, exist_ok=True)

        df = pd.DataFrame().from_dict(data[session])

        plt.figure(figsize=(20, 12), dpi=256)
        for rank in df['rank'].unique():
            df_rank = df[df['rank'] == rank].copy()

            plt.plot(
                df_rank['episode'],
                df_rank['loss'].rolling(window).mean(),
                label=f"Process: {rank}",
            )

        plt.plot([], [], ' ', label=f"Session ID: {session}")

        plt.suptitle(args.env_name, fontsize=18, y=.925)
        plt.ylabel(f'Loss \n ({window}-Episode Rolling Average)')
        plt.xlabel('Episode')
        plt.legend()

        plt.savefig(os.path.join(save_dir, 'loss.png'))


if __name__ == "__main__":
    _ = plot_episode_results(args)
    _ = plot_episode_loss(args)
