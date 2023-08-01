import numpy as np
import matplotlib.pyplot as plt
from hex_world import HexMove

GRID = [
    [
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
    ],
    [
        "5",
        "0",
        "0",
        "X",
        "0",
        "0",
        "X",
        "X",
        "0",
        "0",
    ],
    ["0", "0", "-10", "0", "X", "0", "0", "0", "X", "10"],
]


def get_hex_world_prior():
    p_init = np.ones(30)
    count = 0
    for ri, row in enumerate(GRID):
        for ci, col in enumerate(row):
            if col != "0":
                p_init[ri * 10 + ci] = 0
                count += 1
    p_init /= 30 - count
    return p_init


def policy_list_to_matrix(policy):
    """
    policy maps state to action so is |states| x |actions|
    """
    policy_matrix = np.zeros((30, 6))
    for ri, row in enumerate(policy):
        for ci, col in enumerate(row):
            xi = [h for h in HexMove].index(col)
            policy_matrix[ri * 10 + ci][xi] = 1
    return policy_matrix


def sample_trajectories(transmission_matrix, policy_matrix, p_init, n=1):
    """
    Reward from action in state deterministic
    """
    trajectories = []
    for _ in range(n):
        current_index = np.random.choice(np.arange(30), p=p_init)
        trajectory = []
        count = 0
        while True:
            count += 1
            if current_index == 30 or count > 1000:
                break
            trajectory.append(current_index)
            a = np.random.choice(
                np.arange(policy_matrix.shape[1]), p=policy_matrix[current_index]
            )
            trajectory.append(a)
            # sample next state
            next_state_prob = transmission_matrix[current_index, a, :]
            current_index = np.random.choice(np.arange(31), p=next_state_prob)

        if (
            count < 1000
        ):  # stop inifnite loops. some states get stuck without epsilon soft policy
            trajectories.append(trajectory)

    return trajectories


def first_visit_monte_carlo_policy_eval(
    transsmision_matrix, policy_matrix, reward_matrix, p_init
):
    value_function = np.zeros(31)
    gamma = 0.99
    for _ in range(1000):
        samples = sample_trajectories(transsmision_matrix, policy_matrix, p_init, n=10)
        trajectory = np.array(samples[0])
        G = 0
        returns = [[] for _ in range(31)]  # empty returns per state
        remaining_states = trajectory[::2]
        T_index = int((len(trajectory) + 2) / 2)
        for t in range(T_index - 2, -1, -1):  # count down to 0
            state, action = trajectory[t * 2], trajectory[t * 2 + 1]
            reward = reward_matrix[state, action]
            G = gamma * G + reward
            remaining_states = remaining_states[: len(remaining_states) - 1]
            if state not in remaining_states:  # first visit condition
                returns[state].append(G)
                value_function[state] = sum(returns[state]) / len(returns[state])
    return value_function
