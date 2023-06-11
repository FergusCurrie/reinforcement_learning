import numpy as np


def k_armed_bandit(k):
    """Generate a k_armed_bandit function.

    Args:
        k (int): Number of arms (actions)
    """

    means = np.random.uniform(-10, 10, k)

    def take_action(i):
        """return random reward from normal distribution with mean i
        and unit variance.

        Returns:
            index (i) : index to take mean from
        """
        return np.random.normal(means[i], 1)

    return take_action


def epsilon_greedy(k, epsilon=0.1, n_steps=100):
    take_action = k_armed_bandit(k)
    Q = np.zeros(k)
    Q_selected_count = np.zeros(k)
    total_reward = []
    for i in range(n_steps):
        next_action = np.argmax(Q)
        if np.random.uniform() < epsilon:
            next_action = np.random.randint(0, k)
        reward = take_action(next_action)
        total_reward.append(reward)
        Q_selected_count[next_action] += 1
        count = Q_selected_count[next_action]
        avg = Q[next_action]
        Q[next_action] = (1 - (1 / count)) * avg + (1 / count) * reward
    return np.array(total_reward), take_action


# take_action = k_armed_bandit(5)
