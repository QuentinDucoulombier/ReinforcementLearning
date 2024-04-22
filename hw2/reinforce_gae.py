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

# Define a tuple for storing action log probabilities and value estimates
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Initialize a TensorBoard writer for logging training progress
writer = SummaryWriter("./tb_record_1")

class GAE:
    def __init__(self, gamma, lambda_, num_steps):
        """
        Initialize Generalized Advantage Estimation (GAE) with specified parameters.
        """
        self.gamma = gamma  # Discount factor for future rewards
        self.lambda_ = lambda_  # GAE smoothing parameter
        self.num_steps = num_steps  # Can be set to None for full trajectory calculations

    def __call__(self, rewards, values, next_values, dones):
        """
        Compute the GAE for each time step in a trajectory.
        """
        gae = 0
        returns = []
        advantages = []
        for t in reversed(range(len(rewards))):  # Process from the end of the episode backward
            # Calculate temporal difference error
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            # Update GAE using delta
            gae = delta + self.gamma * self.lambda_ * gae * (1 - dones[t])
            # Store the calculated return and advantage
            returns.insert(0, gae + values[t])
            advantages.insert(0, gae)

        # Normalize advantages to reduce variance in policy gradient updates
        advantages = torch.tensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

class Policy(nn.Module):
    def __init__(self, gamma, lambda_):
        """
        Policy network that estimates both action probabilities and state values.
        """
        super(Policy, self).__init__()
        self.gamma = gamma  # Discount factor for future rewards
        self.lambda_ = lambda_  # GAE smoothing parameter
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.hidden_size = 128
        
        # Network layers
        self.affine1 = nn.Linear(self.observation_dim, self.hidden_size)
        self.action_head = nn.Linear(self.hidden_size, self.action_dim)
        self.value_head = nn.Linear(self.hidden_size, 1)
        
        # Lists for storing actions taken and rewards received
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        """
        Forward pass through the network to produce action probabilities and state value estimates.
        """
        x = F.relu(self.affine1(state))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_value = self.value_head(x)
        return action_prob, state_value

    def select_action(self, state):
        """
        Selects an action based on current state.
        """
        state = torch.from_numpy(state).float()
        probs, state_value = self.forward(state)

        m = Categorical(probs)
        action = m.sample()

        # Save action and its value estimate
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.item()

    def calculate_loss(self):
        """
        Calculate loss for training from stored actions and rewards.
        """
        R = 0
        policy_losses = []
        value_losses = []
        returns = []
        
        # Calculate the next values for GAE computation
        next_values = [s.value for s in self.saved_actions[1:]] + [torch.tensor([0])]
        dones = [False] * (len(self.saved_actions) - 1) + [True]

        # Compute returns and advantages using GAE
        gae = GAE(self.gamma, self.lambda_, None)
        returns, advantages = gae(self.rewards, [s.value for s in self.saved_actions], next_values, dones)
        
        # Calculate the policy and value losses
        for (log_prob, value), advantage in zip(self.saved_actions, advantages):
            policy_losses.append(-log_prob * advantage)  # Policy gradient ascent
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([advantage])))  # Value approximation error
        
        # Combine losses from policy and value function
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        return loss

    def clear_memory(self):
        """
        Clear memory of past actions and rewards after each training episode.
        """
        self.rewards.clear()
        self.saved_actions.clear()

def train(lr=0.01, gamma=0.999, lambda_=0.95):
    """
    Train the policy model using the Adam optimizer and GAE for advantage estimation.
    """
    # Initialize the policy model with specified gamma and lambda values
    model = Policy(gamma, lambda_)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Variable to track the exponential weighted moving average of rewards
    ewma_reward = 0
    
    # Training loop over episodes
    for i_episode in count(1):
        # Reset the environment at the start of each episode
        state = env.reset()
        ep_reward = 0  # Total reward for the episode

        # Run steps within the episode
        for t in range(10000):  # Limit the number of steps per episode
            # Select an action based on the current state
            action = model.select_action(state)
            # Apply the action to the environment
            state, reward, done, _ = env.step(action)
            # Record the reward
            model.rewards.append(reward)
            ep_reward += reward
            # Break if the episode is done
            if done:
                break
        
        # Perform backpropagation
        optimizer.zero_grad()  # Clear previous gradients
        loss = model.calculate_loss()  # Compute loss for the current episode
        loss.backward()  # Perform backpropagation
        optimizer.step()  # Update the model parameters
        model.clear_memory()  # Clear stored data
        
        # Update and log the EWMA of rewards
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))
        
        # Log training information to TensorBoard
        writer.add_scalar('Training/Loss', loss.item(), i_episode)
        writer.add_scalar('Training/Reward', ep_reward, i_episode)
        writer.add_scalar('Training/EWMA_Reward', ewma_reward, i_episode)

        # Check if training goal is achieved
        if ewma_reward > 250:
            # Ensure the directory exists for model saving
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            # Save the model
            torch.save(model.state_dict(), './preTrained/LunarLander_gae_{}.pth'.format(lambda_))
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(ewma_reward, t))
            break

def test(name, n_episodes=10):
    """
    Test the trained model for a set number of episodes.
    """
    # Load the trained model
    model = Policy(0.999, 0.95)  # Initialize the policy with the gamma and lambda used during training
    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    render = True  # Flag to determine whether to render the environment
    max_episode_len = 10000  # Set the maximum length of an episode
    
    # Loop over the number of test episodes
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(max_episode_len+1):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                env.render()  # Visualize the environment
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()

if __name__ == '__main__':
    # Set a random seed for reproducibility
    random_seed = 10  
    lr = 0.02  # Learning rate for the optimizer
    env = gym.make('LunarLander-v2')  # Initialize the environment
    env.seed(random_seed)  
    torch.manual_seed(random_seed)  
    lambda_ = 0.93  # GAE lambda parameter
    # Start training and testing the model
    train(lr, 0.999, lambda_)
    test(f'LunarLander_gae_{lambda_}.pth')
