# Spring 2024, 535514 Reinforcement Learning
# HW1: Policy Iteration and Value iteration for MDPs
       
import numpy as np
import gym

def get_rewards_and_transitions_from_env(env):
    # Get state and action space sizes
    num_spaces = env.observation_space.n
    num_actions = env.action_space.n

    # Intiailize matrices
    R = np.zeros((num_spaces, num_actions, num_spaces))
    P = np.zeros((num_spaces, num_actions, num_spaces))

    # Get rewards and transition probabilitites for all transitions from an OpenAI gym environment
    for s in range(num_spaces):
        for a in range(num_actions):
            for transition in env.P[s][a]:
                prob, s_, r, done = transition
                R[s, a, s_] = r
                P[s, a, s_] = prob
                
    return R, P

def value_iteration(env, gamma=0.9, max_iterations=10**6, eps=10**-3):
    """        
        Run value iteration (You probably need no more than 30 lines)
        
        Input Arguments
        ----------
            env: 
                the target environment
            gamma: float
                the discount factor for rewards
            max_iterations: int
                maximum number of iterations for value iteration
            eps: float
                for the termination criterion of value iteration 
        ----------
        
        Output
        ----------
            policy: np.array of size (500,)
        ----------
        
        TODOs
        ----------
            1. Initialize the value function V(s)
            2. Get transition probabilities and reward function from the gym env
            3. Iterate and improve V(s) using the Bellman optimality operator
            4. Derive the optimal policy using V(s)
        ----------
    """
    num_spaces = env.observation_space.n
    num_actions = env.action_space.n
    
    # Initialize with a random policy
    # Not usefull in VI to initialate with random policy (only use more ressource)?
    #policy = np.array([env.action_space.sample() for _ in range(num_spaces)])
    
    ##### FINISH TODOS HERE #####
    V = np.zeros(num_spaces)  # Initialize V(s) to zeros for all states
    R, P = get_rewards_and_transitions_from_env(env)  # Retrieve rewards (R) and transition probabilities (P) from the environment
    
    for _ in range(max_iterations):
        delta = 0  # Initialize delta to track the maximum change in V(s) for this iteration
        for s in range(num_spaces):  # Iterate over all states
            v_prev = V[s]  # Store the current value of V(s) to compare after update
            # Update V(s) based on the Bellman optimality equation
            # s_ represents the next state s', and a represents each possible action
            V[s] = max([sum([(R[s, a, s_] + gamma * P[s, a, s_] * V[s_]) for s_ in range(num_spaces)]) for a in range(num_actions)])
            # Calculate the maximum difference in V(s) for this iteration to check for convergence
            delta = max(delta, abs(v_prev - V[s]))
        if delta < eps:  # Check if the maximum change in V(s) is below a small threshold, indicating convergence
            break
        print(f"Iteration: {_}, Delta: {delta}")
    
    # Initialize policy with zeros, indicating arbitrary actions for each state initially
    policy = np.zeros(num_spaces, dtype=int)
    for s in range(num_spaces):
        # Update the policy for each state s by choosing the action that maximizes the expected future rewards
        policy[s] = np.argmax([sum([(R[s, a, s_] + gamma * P[s, a, s_] * V[s_]) for s_ in range(num_spaces)]) for a in range(num_actions)])
    #############################


    
    # Return optimal policy    
    return policy

def policy_iteration(env, gamma=0.9, max_iterations=10**6, eps=10**-3):
    """ 
        Run policy iteration (You probably need no more than 30 lines)
        
        Input Arguments
        ----------
            env: 
                the target environment
            gamma: float
                the discount factor for rewards
            max_iterations: int
                maximum number of iterations for the policy evalaution in policy iteration
            eps: float
                for the termination criterion of policy evaluation 
        ----------  
        
        Output
        ----------
            policy: np.array of size (500,)
        ----------
        
        TODOs
        ----------
            1. Initialize with a random policy and initial value function
            2. Get transition probabilities and reward function from the gym env
            3. Iterate and improve the policy
        ----------
    """
    num_spaces = env.observation_space.n
    num_actions = env.action_space.n
    
    # Initialize with a random policy
    policy = np.array([env.action_space.sample() for _ in range(num_spaces)])
    
    ##### FINISH TODOS HERE #####
    # Extract reward and transition matrices from the environment (fonction)
    R, P = get_rewards_and_transitions_from_env(env)

    # Define the policy evaluation function
    def policy_eval(policy, V, R, P, gamma, eps):
        while True:
            delta = 0  # Initialize delta for convergence check
            for s in range(num_spaces):
                v = V[s]
                V[s] = sum([(R[s, policy[s], s_] + gamma * P[s, policy[s], s_] * V[s_]) for s_ in range(num_spaces)])
                delta = max(delta, abs(v - V[s]))  # Update delta with the maximum change in value
            if delta < eps:  # Check for convergence
                break
        return V, delta  # Return the updated value function and the last delta for debugging

    # Begin the policy iteration process
    for iteration in range(max_iterations):
        V = np.zeros(num_spaces)  # Initialize the value function to zeros
        V, last_delta = policy_eval(policy, V, R, P, gamma, eps)  # Perform policy evaluation

        print(f"Iteration {iteration+1}: Last Delta = {last_delta}")  # Print the last delta after policy evaluation

        policy_stable = True  # Assume the policy is stable initially
        for s in range(num_spaces):
            old_action = policy[s]
            # Find the best action by maximizing the expected value
            policy[s] = np.argmax([sum([(R[s, a, s_] + gamma * P[s, a, s_] * V[s_]) for s_ in range(num_spaces)]) for a in range(num_actions)])
            if old_action != policy[s]:  # Check if the action has changed
                policy_stable = False  # Policy is not stable if any action has changed

        if policy_stable:  # If the policy did not change, it has converged to an optimal policy
            print(f"Policy stabilized after {iteration+1} iterations with Last Delta = {last_delta}.")
            break  # Exit the loop since we've found an optimal policy

    #############################


    # Return optimal policy
    return policy

def print_policy(policy, mapping=None, shape=(0,)):
    print(np.array([mapping[action] for action in policy]).reshape(shape))


def run_pi_and_vi(env_name):
    """ 
        Enforce policy iteration and value iteration
    """    
    env = gym.make(env_name)
    print('== {} =='.format(env_name))
    print('# of actions:', env.action_space.n)
    print('# of states:', env.observation_space.n)
    print(env.desc)

    vi_policy = value_iteration(env)
    pi_policy = policy_iteration(env)

    return pi_policy, vi_policy


if __name__ == '__main__':
    # OpenAI gym environment: Taxi-v2 or Taxi-v3
    pi_policy, vi_policy = run_pi_and_vi('Taxi-v3')

    action_map = {0: "S", 1: "N", 2: "E", 3: "W", 4: "P", 5: "D"}
    print('PI-POLICY')
    print_policy(pi_policy, action_map, shape=None)
    print('VI_POLICY')
    print_policy(vi_policy, action_map, shape=None)
    
    # Compare the policies obatined via policy iteration and value iteration
    diff = sum([abs(x-y) for x, y in zip(pi_policy.flatten(), vi_policy.flatten())])        
    print('Discrepancy:', diff)
    



