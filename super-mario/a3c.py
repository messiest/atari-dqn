import os
import time
import csv
from collections import deque
from itertools import count

import numpy as np
import cv2
import gym
from gym import wrappers
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from actor_critic import ActorCritic
from mario_actions import ACTIONS
from mario_wrapper import create_mario_env
from shared_adam import SharedAdam


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def choose_action(model, state, hx, cx):
    self.eval()  # set to eval mode
    logits, _ = self.forward(s)
    prob = F.softmax(logits, dim=1).data
    m = self.distribution(prob)

    return m.sample().numpy()[0]

def train(rank, args, shared_model, counter, lock, optimizer=None, select_sample=True):
    torch.manual_seed(args.seed + rank)

    print(f"Process No: {rank: 3d} | Sampling: {select_sample}")

    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    DoubleTensor = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor
    ByteTensor = torch.cuda.ByteTensor if torch.cuda.is_available() else torch.ByteTensor

    env = create_mario_env(args.env_name)
    # env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], len(ACTIONS))
    if torch.cuda.is_available():
        model.cuda()
    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=LR)

    model.train()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    episode_length = 0
    for t in count():
        if rank == 0:
            env.render()
            if t % args.save_interval == 0 and t > 0:
                for file in os.listdir('checkpoints/'):
                    os.remove(os.path.join('checkpoints', file))
                torch.save(
                    shared_model.state_dict(),
                    os.path.join("checkpoints", f"{env.spec.id}_a3c_params_{t}.pkl")
                )

        if t % args.save_interval == 0 and t > 0 and rank == 1:
            for file in os.listdir('checkpoints/'):
                os.remove(os.path.join('checkpoints', file))
            torch.save(
                shared_model.state_dict(),
                os.path.join("checkpoints", f"{env.spec.id}_a3c_params_{t}.pkl")
            )

        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, 512)).type(FloatTensor)
            hx = Variable(torch.zeros(1, 512)).type(FloatTensor)
        else:
            cx = Variable(cx.data).type(FloatTensor)
            hx = Variable(hx.data).type(FloatTensor)

        values = []
        log_probs = []
        rewards = []
        entropies = []
        reason = ''

        for step in range(args.num_steps):
            episode_length += 1
            # state_inp = Variable(state.unsqueeze(0)).type(FloatTensor)
            # value, logit, (hx, cx) = model((state_inp, (hx, cx)))
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))

            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(-1, keepdim=True)
            entropies.append(entropy)

            if select_sample:
                action = prob.multinomial(num_samples=1).data
            else:
                action = prob.max(-1, keepdim=True)[1].data

            # print("ACTION:", action)

            log_prob = log_prob.gather(-1, Variable(action))

            # print("LOG PROB:", log_prob)

            action_out = ACTIONS[action]
            # print("ACTION OUT", action_out)

            state, reward, done, _ = env.step(action.item())

            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 50), -50)

            with lock:
                counter.value +=1

            if done:
                episode_length = 0
                # env.change_level(0)
                state = env.reset()
                print(f"Process {rank} has completed.")

            env.locked_levels = [False] + [True] * 31
            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            state_inp = Variable(state.unsqueeze(0)).type(FloatTensor)
            value, _, _ = model((state_inp, (hx, cx)))
            R = value.data

        values.append(Variable(R).type(FloatTensor))
        policy_loss = 0
        value_loss = 0
        R = Variable(R).type(FloatTensor)

        gae = torch.zeros(1, 1).type(FloatTensor)

        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - log_probs[i] * Variable(gae).type(FloatTensor) - args.entropy_coef * entropies[i]

        total_loss = policy_loss + args.value_loss_coef * value_loss

        print(f"Process {rank: 2d} loss: {total_loss.item(): 4.2f}")

        optimizer.zero_grad()
        (total_loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)

        optimizer.step()

    print(f"Process {rank} closed.")


def test(rank, args, shared_model, counter):
    torch.manual_seed(args.seed + rank)

    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    DoubleTensor = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor
    ByteTensor = torch.cuda.ByteTensor if torch.cuda.is_available() else torch.ByteTensor

    env = create_mario_env(args.env_name)

    # env = gym.wrappers.Monitor(env, 'playback/', force=True)

    # env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], len(ACTIONS))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)

    reward_sum = 0

    done = True

    save_file = os.getcwd() + '/save/mario_performance.csv'

    title = ['Time', 'Steps', 'Total Reward', 'Episode Length']
    with open(save_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(title)

    start_time = time.time()

    actions = deque(maxlen=4000)
    episode_length = 0
    while True:
        episode_length += 1
        ep_start_time = time.time()

        # shared model sync
        if done:
            model.load_state_dict(shared_model.state_dict())
            with torch.no_grad():
                cx = Variable(torch.zeros(1, 512)).type(FloatTensor)
                hx = Variable(torch.zeros(1, 512)).type(FloatTensor)

        else:
            with torch.no_grad():
                cx = Variable(cx.data).type(FloatTensor)
                hx = Variable(hx.data).type(FloatTensor)

        with torch.no_grad():
            state_inp = Variable(state.unsqueeze(0)).type(FloatTensor)
        value, logit, (hx, cx) = model((state_inp, (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(-1, keepdim=True)[1]

        action_out = ACTIONS[action]  #[0, 0]

        # print("ACTION OUT 2", action_out)

        state, reward, done, _ = env.step(action.item())
        env.render()
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        actions.append(action[0, 0])

        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            stop_time = time.time()
            print("Time: {}, Num Steps: {}, FPS: {:.2f}, Episode Reward: {}, Episode Length: {}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(stop_time - start_time)),
                counter.value,
                counter.value / (stop_time - start_time),
                reward_sum,
                episode_length,
            ))

            data = [
                stop_time - ep_start_time,
                counter.value,
                counter.value / (stop_time - start_time),
                reward_sum,
                episode_length,
            ]

            with open(save_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([data])

            reward_sum = 0
            episode_length = 0
            actions.clear()
            # env.locked_levels = [False] + [True] * 31
            # env.change_level(0)
            state = env.reset()

        state = torch.from_numpy(state)


if __name__ == "__main__":
    pass
