from typing import Tuple, List
from hex_world import HexWorld, HexMove, Hexagon
import numpy as np

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
EAST_POLICY = [[HexMove.EAST for _ in range(10)] for _ in range(3)]
MAX_DEPTH = 1


def loopy_lookahead_recursive(state: Hexagon, depth=0, gamma=1, max_depth=MAX_DEPTH):
    if depth > max_depth:
        return 0

    R = 0
    action = state.policy
    if depth > 0:
        R = state.score
    if R == -1:
        raise Exception("shouldnt be able to move to hole")
    if R != 0:
        return R

    lottery = state.reachable_states(action)
    rollout = 0
    for prob, next_state in lottery:
        if next_state == None:
            rollout += -1 * prob
            r = prob * loopy_lookahead_recursive(state, depth + 1)
            # print(f"_r {depth} {r}")
            rollout += r
            if depth == 0:
                print("cjecl")
        else:
            r = prob * loopy_lookahead_recursive(next_state, depth + 1)
            # print(f"r {depth} {r}")
            rollout += r

    return gamma * rollout


def loopy_lookahead(hw):
    U_oo_lookahead = []
    rows = len(hw.grid)
    cols = len(hw.grid[0])
    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            hexagon = hw.hexagons[i][j]
            if hexagon.score == 0:
                U_oo_lookahead.append(loopy_lookahead_recursive(hexagon))
            else:
                U_oo_lookahead.append(np.inf)
    U_oo_lookahead = np.array(U_oo_lookahead)
    return U_oo_lookahead


def transmission_lookahead_recursive(
    state: int, T, R, policy, depth=0, gamma=1, max_depth=MAX_DEPTH
):
    if depth > max_depth:
        return 0

    reward = 0
    if depth > 0:
        reward = R[state, policy]  # TODO: this is wrong? shouldnt get reward at start
        if reward != 0:
            print(reward)
            return reward

    for next_state_index, next_state_prob in enumerate(T[state, policy]):
        # if next_state_index == 21:
        #     print(next_state_index, next_state_prob)
        if next_state_prob > 0:
            if next_state_index == state:
                reward += next_state_prob * -1
            reward += next_state_prob * transmission_lookahead_recursive(
                next_state_index, T, R, policy, depth + 1
            )
    return gamma * reward


def transmission_lookahead(T, R, hw: HexWorld):
    U_trans_lookahead = []
    rows = len(hw.grid)
    cols = len(hw.grid[0])
    for i in range(rows):
        for j in range(cols):
            hexagon = hw.hexagons[i][j]
            index = i * cols + j
            if hexagon.score == 0:
                U_trans_lookahead.append(
                    transmission_lookahead_recursive(
                        index, T, R, policy=HexMove.EAST.value
                    )
                )
            else:
                U_trans_lookahead.append(np.inf)
    return np.array(U_trans_lookahead)


def policy_eval_system_of_equations(policy, T, R):
    T_policy = T[:, policy, :]
    mask = np.sum(T_policy, axis=-1) == 1
    T_policy_cleaned = T_policy[mask][:, mask]
    U_linear_eq = (
        np.linalg.inv(np.eye(T_policy_cleaned.shape[0]) - T_policy_cleaned)
        @ R[:, policy][..., np.newaxis]
    )
    return U_linear_eq


if __name__ == "__main__":
    hw = HexWorld(grid=GRID, policy=EAST_POLICY)
    T, R = hw.get_mdp()

    print(R.shape)

    # do loopy lookahead with oo implementation
    U_loopy_lookahead = loopy_lookahead(hw)
    print(U_loopy_lookahead.reshape(3, 10))

    # now do lookahead using transsmission matrix
    U_trans_lookahead = transmission_lookahead(T, R, hw)
    print()
    print(U_trans_lookahead.reshape(3, 10))

    # # look at small depth
    # U_lookahead = np.array(
    #     [
    #         transmission_lookahead_recursive(
    #             i, T, R, policy=HexMove.EAST.value, max_depth=0
    #         )
    #         for i in range(30)
    #     ]
    # )
    # print()
    # print(U_lookahead)
