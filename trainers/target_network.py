import random
from itertools import count

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Transition, get_epsilon
from utils.plots import plot_rewards, plot_durations

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


GAMMA = 0.999
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 200
BATCH_SIZE = 128

STEPS = 0


class TargetNetwork(object):
    def __init__(self, env, model, optimizer, memory):
        self.env = env
        self.policy_net = model(env.action_space.n)
        self.target_net = model(env.action_space.n)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optimizer(self.policy_net.parameters())
        self.memory = memory

        self.episode = 0
        self.episode_durations = []
        self.episode_rewards = []

        self.step = 0

        self.env.render()

        print(self.env.spec.id)

    def train(self):
        self.env.reset()
        last_screen = self.env.get_screen().to(DEVICE)
        current_screen = self.env.get_screen().to(DEVICE)
        state = current_screen - last_screen
        episode_reward = 0
        for t in count():
            self.env.render()
            action = self.select_action(state, t)
            a = action.item()
            frame, reward, is_done, info = self.env.step(a)

            episode_reward += reward
            reward = torch.tensor([reward], device=DEVICE)
            last_screen = current_screen
            current_screen = self.env.get_screen().to(DEVICE)
            if not is_done:
                next_state = current_screen - last_screen
            else:
                next_state = None
            self.memory.push(state, action, next_state, reward)
            state = next_state
            self.optimize_model()
            if is_done:
                self.episode_rewards.append(episode_reward)
                self.episode_durations.append(t)
                print(f"Episode {self.episode+1: 5d} | Duration {t+1: 4d} | Reward {episode_reward: 3.2f}")
                self.episode += 1
                # plot_durations(self.env.spec.id, self.episode_durations)
                plot_rewards(self.env.spec.id, self.episode_rewards)
                break

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=DEVICE,
            dtype=torch.uint8
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(DEVICE)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = F.smooth_l1_loss(
            state_action_values,
            expected_state_action_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()

        self.policy_net.clip()  # clip reward values, inplace
        self.optimizer.step()

    def select_action(self, state, step):
        rand = random.random()
        epsilon = get_epsilon(step)
        # print("SELECT ACTION: STATE", state)
        if rand > epsilon:
            with torch.no_grad():
                state = state.to(DEVICE)
                return self.policy_net(state).max(1)[1].view(1, 1).to(DEVICE)
        else:
            n = self.policy_net.n_actions
            return torch.tensor(
                [[random.randrange(n)]],
                device=DEVICE,
                dtype=torch.long,
            )


if __name__ == "__main__":
    model = AtariDQN
