import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import collections
import tqdm

Observation = collections.namedtuple('Observation', ('state', 'action', 'reward', 'next_state', 'done'))

class DQN(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, 48)
        self.fc2 = nn.Linear(48, 48)
        self.fc3 = nn.Linear(48, n_outputs)
    def forward(self, x):        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)/10
        return x

class DQNSolver:
    GAMMA = 0.9
    BATCH_SIZE = 64
    REPLAY_MEMORY = 10000
    MIN_REPLAY_MEMORY = 1000
    def __init__(self, n_inputs, n_outputs):
      self.device = "cuda:0" if torch.cuda.is_available() else 'cpu'
      self.dqn = DQN(n_inputs, n_outputs).to(self.device)
      self.criterion  = torch.nn.MSELoss()
      self.num_actions = n_outputs
      self.opt = torch.optim.Adam(self.dqn.parameters(), lr = 0.001)
      self.replay_memory = collections.deque([], maxlen = 10000)
  
    def choose_action(self, state, epsilon):
        if (np.random.random() <= epsilon):
            return np.random.randint(self.num_actions)
        else:
            with torch.no_grad():
                return torch.argmax(self.dqn(state.to(self.device))).item()
    
    def remember(self, observation):
        #add past actions to deque memory
        self.replay_memory.append(observation)
        
    def replay(self):
        #take a random minibatch from memory - largeset minibatch size is given by batch_size
        #for each minibatch, y is the dqn action from state
        
        #reward is +1 for surviving another tick, +0 for not living
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY:
            return None
        y_batch, y_target_batch = [], []
        minibatch = random.sample(self.replay_memory, self.BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            y = self.dqn(state.to(self.device))
            y_target = y.clone().detach()
            with torch.no_grad():
                y_target[0][action] = reward/10 if done else reward + self.GAMMA * torch.max(self.dqn(next_state.to(self.device))[0])
            y_batch.append(y[0])
            y_target_batch.append(y_target[0])
        
        y_batch = torch.cat(y_batch)
        y_target_batch = torch.cat(y_target_batch)
        
        self.opt.zero_grad()
        loss = self.criterion(y_batch, y_target_batch)
        loss.backward()
        self.opt.step()        
        
        return loss.item()

    def distill(self, n_bins=[4,4,4,4], lims=[[-1.2, 1.2], [-2, 2], [-0.2094395, 0.2094395], [-3, 3]]):
        '''
        params:
            n_bins: array of number of bins for "Position", "Velocity", "Angle", and "AngVelocity"
            lims: array of [lower, upper] limits for each state property
        returns:
            ret: 4-D array with shape n_bins, entries go from lower->upper bin, 1=right, 0=left
        '''
        steps = [(l[1] - l[0]) / n for l, n in zip(lims, n_bins)]
        state_init = torch.Tensor([l[0] + 0.5*step for l, step in zip(lims, steps)]) #init to midpoints of lowest bins
        curr_state = state_init.clone()
        ret = torch.ones(n_bins)
        for i in tqdm.tqdm(range(n_bins[0])):
            for j in tqdm.tqdm(range(n_bins[1])):
                for k in range(n_bins[2]):
                    for h in range(n_bins[3]):
                        ret[i,j,k,h] = self.choose_action(curr_state, epsilon=0.)
                        curr_state[3] += steps[3] #increment by step
                    curr_state[2] += steps[2]
                    curr_state[3] = state_init[3] #reset state
                curr_state[1] += steps[1]
                curr_state[2] = state_init[2]
            curr_state[0] += steps[0]
            curr_state[1] = state_init[1]
        return ret


if __name__=="__main__":
    env = gym.make("CartPole-v1")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)

    epsilon_start = 1
    epsilon_end = 0.1
    epsilon_decay = 0.99
    epsilon = epsilon_start

    max_episodes = 1000
    episode = 0

    episode_runs = []
    episode_losses = []
    while episode < max_episodes:
        episode += 1
        state = env.reset()
        state = torch.tensor(np.reshape(state[0], [1, observation_space]))
        step = 0
        episode_loss = 0
        while True:
            action = dqn_solver.choose_action(state,epsilon)
            state_next, reward, terminal, truncated, info = env.step(action)
            reward = reward if not terminal else 0
            state_next = torch.tensor(np.reshape(state_next, [1, observation_space]))
            dqn_solver.remember(Observation(state, action, reward, state_next, terminal))
            loss = dqn_solver.replay()
            
            if loss is not None:
                episode_loss += loss
                state = state_next
                step += 1

            if terminal or truncated:
                if step > 0:
                    print("Run: " + str(episode) + ", score: " + str(step) + ", episode_loss: " + str(episode_loss/step))
                    episode_runs.append(step)
                    episode_losses.append(episode_loss/step)
                    epsilon = epsilon * epsilon_decay
                    epsilon = min(epsilon, epsilon_end)
                break
          