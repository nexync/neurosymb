import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import collections
import math
import tqdm

from CartpoleAgent import FOLCartpoleAgent

Observation = collections.namedtuple("Observation", ("Position", "Velocity", "Angle", "AngVelocity"))
Transition = collections.namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))

EPSILON_START = 1.0
EPSILON_END = 0.1
WARMUP_EPISODES = 500
NUM_EPISODES = 10_000

n_bin_args = {
    "n_pos": 10,
    "n_vel": 10,
    "n_ang": 10,
    "n_angvel": 10
}

limits = {
    "Position": [-1.2, 1.2],
    "Velocity": [-2, 2],
    "Angle": [-0.2094395, 0.2094395],
    "AngVelocity": [-3, 3]
}

agent = FOLCartpoleAgent(n_bin_args, n_nodes = 10, limits = limits)
env = gym.make("CartPole-v1")

def train():
    epsilon = EPSILON_START
    episode_num = 0
    episode_runs = []
    episode_losses = []
    for episode in tqdm.tqdm(range(NUM_EPISODES)):
        total_loss, step = 0,0
        state, info = env.reset()
        while True:
            if np.random.random() > epsilon:
                action = agent.sample_random_action()
            else:
                action = agent.get_action(state)
            next_state, reward, terminal, truncated, info = env.step(action)
            reward = reward/10 if not terminal else 0
            agent.remember(Transition(state, action, next_state, reward, terminal))
            loss = agent.optimize()
            state = next_state

            if loss is not None:
                total_loss += loss
                step += 1
            
            if terminal or truncated:
                if step > 0:
                    print("Run: " + str(episode) + ", score: " + str(step) + ", episode_loss: " + str(total_loss/step))
                    episode_runs.append(step)
                    episode_losses.append(total_loss/step)
                    epsilon -= (EPSILON_START - EPSILON_END)/WARMUP_EPISODES
                    epsilon = min(epsilon, EPSILON_END)
                break