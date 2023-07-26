# import pytest
import numpy as np
from hex_world import HexMove, HexWorld
from policy_evaluation import transmission_lookahead, loopy_lookahead

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


# def test_transmission_lookahead_recursive_small_max_depth():
#     grid = [["10", "0", "0", "0", "0", "10"]]
#     policy = [[HexMove.EAST for _ in range(10)]]
#     hw = HexWorld(grid=grid, policy=policy)
#     T, R = hw.get_mdp()
#     U = transmission_lookahead(T, R, hw)
#     print(R)
#     print(U)


# def test_loopy_lookahead_recursive_small_max_depth():
#     grid = [["10", "0", "0", "0", "0", "10"], ["X", "X", "X", "X", "X", "X"]]
#     policy = [[HexMove.EAST for _ in range(10)], [HexMove.EAST for _ in range(10)]]
#     hw = HexWorld(grid=grid, policy=policy)
#     T, R = hw.get_mdp()
#     U = loopy_lookahead(hw)
#     print(R)
#     print(U)


def test_transmission_and_loopy_equal_small_problem():
    grid = [["10", "0", "0", "0", "0", "10"], ["X", "X", "X", "X", "X", "X"]]
    policy = [[HexMove.EAST for _ in range(10)], [HexMove.EAST for _ in range(10)]]
    hw = HexWorld(grid=grid, policy=policy)
    T, R = hw.get_mdp()
    Ut = transmission_lookahead(T, R, hw)
    Ul = loopy_lookahead(hw)
    assert np.allclose(Ut, Ul)


def test_transmission_and_loopy_equal_large_problem():
    hw = HexWorld(grid=GRID, policy=EAST_POLICY)
    T, R = hw.get_mdp()
    Ut = transmission_lookahead(T, R, hw)
    Ul = loopy_lookahead(hw)
    assert np.allclose(Ut, Ul)
