import numpy as np
from typing import Callable


def k_armed_bandit_problem(k: int):
    """Generate a k_armed_bandit function.

    Args:
        k (int): Number of arms (actions)
    """

    means = np.random.normal(0, 1, k)

    def take_action(i):
        """return random reward from normal distribution with mean i
        and unit variance.

        Returns:
            index (i) : index to take mean from
        """
        return np.random.normal(means[i], 1)

    return take_action


def incremental_epsilon_greedy(
    k: int, epsilon: float, n_steps: int, take_action: Callable[[int], int]
):
    """Run an epsilon greedy algorithm on a k_armed_bandit.

    Uses an incremental estimation of the mean reward from each bandit.

    Args:
        k (_type_): number of bandits
        epsilon (float, optional): probability of random action. Defaults to 0.1.
        n_steps (int, optional): length of episode. Defaults to 100.
        take_action (Callable): function for probalistic reward from bandit

    Returns:
        _type_: an array of rewards per step
    """
    Q = np.zeros(k)
    Q_selected_count = np.zeros(
        k
    )  # hold counts of number times each action selected for incremental mean
    total_reward = []
    for i in range(n_steps):
        # epsilon greedy action selection
        next_action = np.argmax(Q)
        if np.random.uniform() < epsilon:
            next_action = np.random.randint(0, k)
        reward = take_action(next_action)

        # updates
        total_reward.append(reward)
        Q_selected_count[next_action] += 1
        count = Q_selected_count[next_action]

        # incremental mean
        avg = Q[next_action]
        Q[next_action] = (1 - (1 / count)) * avg + (1 / count) * reward
    return np.array(total_reward)


def epsilon_greedy(
    k: int, epsilon: float, n_steps: int, take_action: Callable[[int], int]
):
    """Run an epsilon greedy algorithm on a k_armed_bandit.

    Args:
        k (_type_): number of bandits
        epsilon (float, optional): probability of random action. Defaults to 0.1.
        n_steps (int, optional): length of episode. Defaults to 100.
        take_action (Callable): function for probalistic reward from bandit

    Returns:
        _type_: an array of rewards per step
    """
    Q = np.zeros(k)
    reward_from_bandit_memory: list[list[int]] = [[] for _ in range(k)]

    total_reward = []
    for _ in range(n_steps):
        # greedy selection, epislon chance of random
        next_action: int = np.argmax(Q)
        if np.random.uniform() < epsilon:
            next_action = np.random.randint(0, k)

        # find reward from action and add to memory
        reward: int = take_action(next_action)
        total_reward.append(reward)

        # update memory
        reward_from_bandit_memory[next_action].append(reward)

        # Q of action is mean of rewards from that action
        Q[next_action] = sum(reward_from_bandit_memory[next_action]) / len(
            reward_from_bandit_memory[next_action]
        )
    return np.array(total_reward)


def upper_confidence_bound_action_selection(
    k: int, c: float, n_steps: int, take_action: Callable[[int], int]
):
    """Run the ucb action selection algorithm on a k_armed_bandit.

    Uses an incremental estimation of the mean reward from each bandit.

    Args:
        k (_type_): number of bandits
        epsilon (float, optional): probability of random action. Defaults to 0.1.
        n_steps (int, optional): length of episode. Defaults to 100.
        take_action (Callable): function for probalistic reward from bandit

    Returns:
        _type_: an array of rewards per step
    """
    Q = np.zeros(k)
    Q_selected_count = np.zeros(
        k
    )  # hold counts of number times each action selected for incremental mean
    total_reward = []
    for i in range(n_steps):
        # UCB action selection
        next_action = np.argmax(Q + c * np.sqrt(np.log(i + 1) / Q_selected_count))

        reward = take_action(next_action)

        # updates
        total_reward.append(reward)
        Q_selected_count[next_action] += 1
        count = Q_selected_count[next_action]

        # incremental mean
        avg = Q[next_action]
        Q[next_action] = (1 - (1 / count)) * avg + (1 / count) * reward
    return np.array(total_reward)


def incremental_gradient_bandit(
    k: int, alpha: float, n_steps: int, take_action: Callable[[int], int]
):
    """Run a gradient based bandit algorithm on a k_armed_bandit.

    Uses an incremental estimation of the mean reward from each bandit.

    Args:
        k (_type_): number of bandits
        epsilon (float, optional): probability of random action. Defaults to 0.1.
        n_steps (int, optional): length of episode. Defaults to 100.
        take_action (Callable): function for probalistic reward from bandit

    Returns:
        _type_: an array of rewards per step
    """
    preference = np.zeros(k)
    A_selected_count = np.zeros(k)
    R_mean = np.zeros(k)
    total_reward = []
    for i in range(n_steps):
        # find probalistic preference for each action, with softmax
        prob_preference = np.exp(preference) / np.sum(np.exp(preference))
        next_action = np.argmax(prob_preference)
        reward = take_action(next_action)

        # updates
        total_reward.append(reward)
        A_selected_count[next_action] += 1

        # update preference
        preference[next_action] = preference[next_action] + alpha * (
            reward - R_mean[next_action]
        ) * (1 - prob_preference[next_action])
        for a in range(k):
            if a != next_action:
                preference[a] = (
                    preference[a] - alpha * (reward - R_mean[a]) * prob_preference[a]
                )

        # incremental mean
        count = A_selected_count[next_action]
        R_mean[next_action] = (1 - (1 / count)) * R_mean[next_action] + (
            1 / count
        ) * reward
        # Q[next_action] = (1 - (1 / count)) * avg + (1 / count) * reward
    return np.array(total_reward)
