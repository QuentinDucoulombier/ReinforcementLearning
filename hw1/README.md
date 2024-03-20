# Reinforcement Learning hw1 Assignments

This repository contains Python scripts for solving specific problems in reinforcement learning, specifically focusing on policy and value iteration for Markov Decision Processes (MDPs) and performing sanity checks with datasets from D4RL.

## Files

### `policy_and_value_iteration.py`

- **Purpose**: Implements policy iteration and value iteration algorithms for MDPs. Designed to solve problems 4
- **Key Functions**:
  - `value_iteration(env, gamma, max_iterations, eps)`: Performs value iteration on a given environment `env` with specified parameters.
  - `policy_iteration(env, gamma, max_iterations, eps)`: Implements policy iteration on a given environment `env` with specified parameters.
  - `run_pi_and_vi(env_name)`: Helper function to execute both policy iteration and value iteration on a specified environment.
- **Usage**: Run this script to test policy and value iteration on environments like Taxi-v3 from OpenAI's gym.

### `d4rl_sanity_check.py`

- **Purpose**: Provides a sanity check using the D4RL datasets to ensure that the environments and data loading mechanisms work as expected.(Problem 5)
- **Key Functions**:
  - The script creates an environment (in this case, `hopper-medium-v0` from D4RL), interacts with it using a sample action, and demonstrates how to access the dataset associated with the environment.


## Dependencies

- Python 3.8+
- `gym`
- `numpy`
- `d4rl` - For the D4RL datasets and environments.

## Running the Scripts

To run the policy and value iteration script on the Taxi-v3 environment:

```bash
python3 policy_and_value_iteration.py
```

To perform the sanity check with D4RL:

```bash
python3 d4rl_sanity_check.py
```
