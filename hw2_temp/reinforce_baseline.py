# Spring 2024, 535514 Reinforcement Learning
# HW2: REINFORCE and baseline

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
import torch.optim.lr_scheduler as Scheduler
from torch.utils.tensorboard import SummaryWriter

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Define a tensorboard writer
writer = SummaryWriter("./tb_record_1")
        
class Policy(nn.Module):
    """
    Implement a combined network for policy and value functions with a shared layer.
    """
    def __init__(self):
        super(Policy, self).__init__()
        # Check the type of action space to configure network outputs correctly
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128
        self.double()
        
        # Define the shared, action, and value layers in the network
        self.shared_layer = nn.Linear(self.observation_dim, self.hidden_size)
        self.action_layer = nn.Linear(self.hidden_size, self.action_dim)
        self.value_layer = nn.Linear(self.hidden_size, 1)
        
        # Lists for storing actions taken and rewards received
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        """
        Define the forward pass for both the policy and value outputs.
        """
        # Pass the input state through the shared layer and apply ReLU activation
        x = F.relu(self.shared_layer(state))
        # Compute the action probabilities using softmax
        action_prob = F.softmax(self.action_layer(x), dim=-1)
        # Compute the value estimate
        state_value = self.value_layer(x)

        return action_prob, state_value

    def select_action(self, state):
        """
        Select an action based on current policy and given state.
        """
        # Convert the state to a PyTorch tensor and process through the network
        state = torch.from_numpy(state).float()
        probs, state_value = self.forward(state)
        m = Categorical(probs)
        action = m.sample()

        # Save the action's log probability and its estimated value
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()

    def calculate_loss(self, gamma=0.99):
        """
        Calculate the combined loss for the policy gradient update.
        """
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []
        returns = []

        # Calculate the discounted returns
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).float()

        # Normalize the returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        # Calculate policy and value losses
        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.detach()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).float()))

        # Sum the policy and value losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        return loss

    def clear_memory(self):
        """
        Clear the saved actions and rewards after each episode.
        """
        self.rewards.clear()
        self.saved_actions.clear()

def train(lr=0.01):
    """
    Train the model using Stochastic Gradient Descent (SGD) and backpropagation to update both the policy and value network.
    """
    # Create a policy model instance and an optimizer using Adam algorithm
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Optional: Define a learning rate scheduler that decreases the learning rate periodically
    scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    # Exponentially Weighted Moving Average (EWMA) reward for monitoring progress
    ewma_reward = 0
    
    # Run training loop indefinitely
    for i_episode in count(1):
        # Initialize environment and episode reward
        state = env.reset()
        ep_reward = 0
        t = 0

        # Update the learning rate according to the scheduler
        scheduler.step()
        
        # Run a maximum of 9999 steps per episode to prevent infinite loops
        for t in range(10000):
            # Select an action based on the current state and apply it to the environment
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            # Store the reward
            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # Reset gradients, calculate loss, and perform a backward pass for network updates
        optimizer.zero_grad()
        loss = model.calculate_loss()
        loss.backward()
        optimizer.step()
        model.clear_memory()

        # Update the EWMA reward and log the training progress
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

        # Log metrics to TensorBoard
        writer.add_scalar('Training/Loss', loss.item(), i_episode)
        writer.add_scalar('Training/Reward', ep_reward, i_episode)
        writer.add_scalar('Training/EWMA_Reward', ewma_reward, i_episode)
        writer.add_scalar('Training/Length', t, i_episode)  
        writer.add_scalar('Training/Learning_Rate', optimizer.param_groups[0]['lr'], i_episode)

        # Check if the problem is solved (threshold is arbitrary and problem-specific)
        if ewma_reward > 120:
            # Save the trained model
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), './preTrained/LunarLander_{}.pth'.format(lr))
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(ewma_reward, t))
            break

def test(name, n_episodes=10):
    """
    Test the learned policy on a specified number of episodes to evaluate its performance.
    """     
    # Load the pre-trained model
    model = Policy()
    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    
    # Set to render the environment visually
    render = True
    max_episode_len = 10000
    
    # Evaluate the model on a series of episodes
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        running_reward = 0
        for t in range(max_episode_len + 1):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            # Optionally render the environment
            if render:
                env.render()
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()

if __name__ == '__main__':
    # Set a random seed for reproducibility
    random_seed = 10
    lr = 0.02
    # Initialize the environment
    env = gym.make('LunarLander-v2')
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    # Begin training
    train(lr)
    # Test the trained model
    test(f'LunarLander_{lr}.pth')

        