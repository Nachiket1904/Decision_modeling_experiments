import numpy as np
from mdptoolbox.mdp import ValueIteration

# Define the states: combinations of bread and topping
STATES = [
    ('white', 'lettuce'),
    ('white', 'tomato'),
    ('white', 'avocado'),
    ('whole wheat', 'lettuce'),
    ('whole wheat', 'tomato'),
    ('whole wheat', 'avocado'),
    ('rye', 'lettuce'),
    ('rye', 'tomato'),
    ('rye', 'avocado')
]

# Define the actions: combinations of bread and filling
ACTIONS = [
    ('white', 'turkey'),
    ('white', 'ham'),
    ('white', 'cheese'),
    ('whole wheat', 'turkey'),
    ('whole wheat', 'ham'),
    ('whole wheat', 'cheese'),
    ('rye', 'turkey'),
    ('rye', 'ham'),
    ('rye', 'cheese')
]

# Define the transition function
def transition_function(state, action):
    bread, _ = state
    new_bread, filling = action
    if filling == 'turkey':
        new_topping = 'lettuce'
    elif filling == 'ham':
        new_topping = 'tomato'
    else:
        new_topping = 'avocado'
    return (new_bread, new_topping)

# Define the reward function
def reward_function(state, action):
    bread, topping = state
    _, filling = action
    if bread == 'white' and filling == 'turkey' and topping == 'avocado':
        return 10  # Perfect sandwich
    elif bread == 'whole wheat' and filling == 'ham' and topping == 'tomato':
        return 8  # Good sandwich
    elif bread == 'rye' and filling == 'cheese' and topping == 'avocado':
        return 9  # Great sandwich
    else:
        return 0  # Unsatisfactory sandwich

# Create the transition and reward matrices
num_states = len(STATES)
num_actions = len(ACTIONS)
transitions = np.zeros((num_states, num_states, num_actions))
rewards = np.zeros((num_states, num_actions))

for state_idx, state in enumerate(STATES):
    for action_idx, action in enumerate(ACTIONS):
        new_state = transition_function(state, action)
        new_state_idx = STATES.index(new_state)
        transitions[state_idx, new_state_idx, action_idx] = 1
        rewards[state_idx, action_idx] = reward_function(state, action)

# Ensure that transitions and rewards have the correct dimensions
assert transitions.shape == (num_states, num_states, num_actions), f"Transitions matrix has incorrect shape: {transitions.shape}"
assert rewards.shape == (num_states, num_actions), f"Rewards matrix has incorrect shape: {rewards.shape}"

# Define the MDP parameters
discount = 0.9  # Discount factor for future rewards
epsilon = 0.001  # Convergence criterion
max_iter = 1000  # Maximum number of iterations

# Create the MDP object using ValueIteration
sandwich_mdp = ValueIteration(transitions, rewards, discount, max_iter=max_iter, epsilon=epsilon)

# Solve the MDP using value iteration
sandwich_mdp.run()

# Extract the optimal policy
policy = np.array(sandwich_mdp.policy)

# Print the optimal policy
print("Optimal Policy:")
for state_idx, state in enumerate(STATES):
    action_idx = policy[state_idx]
    action = ACTIONS[action_idx]
    print(f"For state {state}, the optimal action is to choose {action}")

