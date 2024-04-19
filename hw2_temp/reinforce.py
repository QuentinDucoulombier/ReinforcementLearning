# Import necessary libraries and modules
import os
import gym
from itertools import count
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

# Define a tuple for storing log probabilities of actions
SavedAction = namedtuple('SavedAction', ['log_prob'])

# Initialize a tensorboard writer to log training data
writer = SummaryWriter("./tb_record_1")

class Policy(nn.Module):
    """
    Define a policy network that outputs both action probabilities and state values, which can be used for reinforcement learning.
    """
    def __init__(self):
        super(Policy, self).__init__()
        # Determine if the action space is discrete and get the dimensions of observation and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128
        self.double()
        
        # Neural network layers
        self.affine1 = nn.Linear(self.observation_dim, self.hidden_size)
        self.action_head = nn.Linear(self.hidden_size, self.action_dim)
        
        # Lists to store actions taken and rewards received
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        """
        Perform a forward pass of the network to get action probabilities.
        """
        x = F.relu(self.affine1(state))
        action_prob = F.softmax(self.action_head(x), dim=0)
        return action_prob

    def select_action(self, state):
        """
        Selects an action for a given state using the policy network, and stores the log probability of the action.
        """
        state = torch.from_numpy(state).float()
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_actions.append(SavedAction(m.log_prob(action)))
        return action.item()

    def calculate_loss(self, gamma=0.999):
        """
        Calculate the policy gradient loss using the saved actions and rewards.
        """
        R = 0
        policy_losses = []
        returns = []
        # Calculate discounted reward
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        for (log_prob,), R in zip(self.saved_actions, returns):
            policy_losses.append(-log_prob * R)
        loss = torch.stack(policy_losses).sum()
        return loss

    def clear_memory(self):
        """
        Reset the memory of actions and rewards after each episode.
        """
        self.rewards.clear()
        self.saved_actions.clear()

def train(lr=0.01):
    """
    Train the policy network using the REINFORCE algorithm.
    """
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ewma_reward = 0  # Exponentially weighted moving average of the reward

    for i_episode in count(1):
        state = env.reset()
        ep_reward = 0

        for t in range(10000):  # Limit max steps per episode to prevent infinite loop
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        optimizer.zero_grad()
        loss = model.calculate_loss()
        loss.backward()
        optimizer.step()
        model.clear_memory()

        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print(f'Episode {i_episode}\tlength: {t}\treward: {ep_reward}\tewma reward: {ewma_reward}')

        # Logging to TensorBoard
        writer.add_scalar('Training/Loss', loss.item(), i_episode)
        writer.add_scalar('Training/Reward', ep_reward, i_episode)
        writer.add_scalar('Training/EWMA_Reward', ewma_reward, i_episode)

        # Stop condition if problem is solved
        if ewma_reward > env.spec.reward_threshold:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), './preTrained/CartPole_{}.pth'.format(lr))
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(ewma_reward, t))
            break

def test(name, n_episodes=10):
    """
    Test the trained model on a set number of episodes to evaluate its performance.
    """
    model = Policy()
    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    render = True
    
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        running_reward = 0
        
        for t in range(10001):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                env.render()
            if done:
                break
        print(f'Episode {i_episode}\tReward: {running_reward}')
    env.close()

if __name__ == '__main__':
    random_seed = 10
    lr = 0.02
    env = gym.make('CartPole-v0')
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    train(lr)
    test(f'CartPole_{lr}.pth')
