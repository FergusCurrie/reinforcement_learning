"""Implementation of hexworld problem. 

Hexworld is a discrete sequential decision problem. The agent attempts to traverse the hexagonal grid 
to reach one of the scored terminal states. Most tiles give no reward. Moves are probalistic. A policy 
maps a move to a tile. 
"""

import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from typing import Optional

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
    [
        "0",
        "0",
        "-10",
        "0",
        "X",
        "0",
        "0",
        "0",
        "X",
        "10",
    ],
]


class HexMove(Enum):
    NORTH_WEST = 0
    NORTH_EAST = 1
    EAST = 2
    SOUTH_EAST = 3
    SOUTH_WEST = 4
    WEST = 5


move2clockwise_move = {
    HexMove.NORTH_WEST: HexMove.NORTH_EAST,
    HexMove.NORTH_EAST: HexMove.EAST,
    HexMove.EAST: HexMove.SOUTH_EAST,
    HexMove.SOUTH_EAST: HexMove.SOUTH_WEST,
    HexMove.SOUTH_WEST: HexMove.WEST,
    HexMove.WEST: HexMove.NORTH_WEST,
}

move2anti_clockwise_move = {
    HexMove.NORTH_WEST: HexMove.WEST,
    HexMove.WEST: HexMove.SOUTH_WEST,
    HexMove.SOUTH_WEST: HexMove.SOUTH_EAST,
    HexMove.SOUTH_EAST: HexMove.EAST,
    HexMove.EAST: HexMove.NORTH_EAST,
    HexMove.NORTH_EAST: HexMove.NORTH_WEST,
}


class Hexagon:
    def __init__(
        self, score: int, blank: bool = False, policy: Optional[HexMove] = None
    ):
        self.score = score
        self.blank = blank
        self.north_west = None
        self.north_east = None
        self.east = None
        self.south_east = None
        self.south_west = None
        self.west = None
        self.policy = policy

    def reachable_states(self, move: HexMove):
        """Returns a lottery of reachable states from this hexagon"""
        return [
            (1, self.get_next_hexagon(move)),
        ]
        return [
            (0.15, self.get_next_hexagon(move2clockwise_move[move])),
            (0.7, self.get_next_hexagon(move)),
            (0.15, self.get_next_hexagon(move2anti_clockwise_move[move])),
        ]

    # def fuzzy_move(self, move: HexMove) -> HexMove:
    #     r = np.random.random()
    #     if r < 0.15:
    #         # return anticlockwise move
    #         return move2anti_clockwise_move[move]
    #     elif r > 0.85:
    #         # return clockwise move
    #         return move2clockwise_move[move]
    #     else:
    #         # return move
    #         return move

    def get_next_hexagon(self, move: HexMove):
        if move == HexMove.NORTH_WEST:
            return self.north_west
        elif move == HexMove.NORTH_EAST:
            return self.north_east
        elif move == HexMove.EAST:
            return self.east
        elif move == HexMove.SOUTH_EAST:
            return self.south_east
        elif move == HexMove.SOUTH_WEST:
            return self.south_west
        elif move == HexMove.WEST:
            return self.west
        else:
            raise ValueError("Invalid move")

    # def move(self, move: HexMove):
    #     fuzzy_move = self.fuzzy_move(move)
    #     return self.get_next_hexagon(fuzzy_move)


class HexWorld:
    def __init__(self, grid: list, policy: list):
        self.position = [0, 0]
        self.hexagons = []
        self.grid = grid
        # init hexagons
        for row_index, row in enumerate(grid):
            hexagon_row = []
            for col_index, hexagon in enumerate(row):
                if hexagon != "X":
                    hexagon = Hexagon(
                        score=int(hexagon), policy=policy[row_index][col_index]
                    )
                    hexagon_row.append(hexagon)
                else:
                    hexagon = Hexagon(
                        score=-1, blank=True, policy=policy[row_index][col_index]
                    )
                    hexagon_row.append(hexagon)
            self.hexagons.append(hexagon_row)

        # link hexagons
        for row_index, row in enumerate(grid):
            for col_index, hexagon_row in enumerate(row):
                if hexagon_row != "X":
                    hexagon = self.hexagons[row_index][col_index]

                    if col_index + 1 < len(row):
                        if not self.hexagons[row_index][col_index + 1].blank:
                            hexagon.east = self.hexagons[row_index][col_index + 1]

                    if col_index - 1 >= 0:
                        if not self.hexagons[row_index][col_index - 1].blank:
                            hexagon.west = self.hexagons[row_index][col_index - 1]

                    if row_index % 2 == 0:
                        # north east
                        if row_index - 1 >= 0:
                            if not self.hexagons[row_index - 1][col_index].blank:
                                hexagon.north_east = self.hexagons[row_index - 1][
                                    col_index
                                ]
                        # north west
                        if row_index - 1 >= 0 and col_index - 1 >= 0:
                            if not self.hexagons[row_index - 1][col_index - 1].blank:
                                hexagon.north_west = self.hexagons[row_index - 1][
                                    col_index - 1
                                ]

                        # south east
                        if row_index + 1 < len(grid):
                            if not self.hexagons[row_index + 1][col_index].blank:
                                hexagon.south_east = self.hexagons[row_index + 1][
                                    col_index
                                ]

                        # south west
                        if row_index + 1 < len(grid) and col_index - 1 >= 0:
                            if not self.hexagons[row_index + 1][col_index - 1].blank:
                                hexagon.south_west = self.hexagons[row_index + 1][
                                    col_index - 1
                                ]

                    if row_index % 2 != 0:
                        # north east
                        if row_index - 1 >= 0 and col_index + 1 < len(row):
                            if not self.hexagons[row_index - 1][col_index + 1].blank:
                                hexagon.north_east = self.hexagons[row_index - 1][
                                    col_index + 1
                                ]
                        # north west
                        if row_index - 1 >= 0:
                            if not self.hexagons[row_index - 1][col_index].blank:
                                hexagon.north_west = self.hexagons[row_index - 1][
                                    col_index
                                ]

                        # south east
                        if row_index + 1 < len(grid) and col_index + 1 < len(row):
                            if not self.hexagons[row_index + 1][col_index + 1].blank:
                                hexagon.south_east = self.hexagons[row_index + 1][
                                    col_index + 1
                                ]

                        # south west
                        if row_index + 1 < len(grid) and col_index - 1 > 0:
                            if not self.hexagons[row_index + 1][col_index - 1].blank:
                                hexagon.south_west = self.hexagons[row_index + 1][
                                    col_index - 1
                                ]

    def get_mdp_transition_matrix(self):
        """Get transsmisions matrix corresponding to the hexagon world.

        Transition matrix maps state and action to probability of each state.
        Should be |states| x |actions| x |states|
        """
        states = [state for row in self.hexagons for state in row]  # if not state.blank
        num_states = len(states)
        T = np.zeros((num_states, 6, num_states))
        for row_index, row in enumerate(self.hexagons):
            for col_index, hexagon in enumerate(row):
                if hexagon.blank:
                    continue
                for move in HexMove:
                    lottery = hexagon.reachable_states(move)
                    for prob, next_state in lottery:
                        from_index = states.index(hexagon)
                        if next_state == None:
                            to_index = from_index
                        else:
                            to_index = states.index(next_state)
                        T[from_index, move.value, to_index] += prob
        return T

    def get_mdp_reward_matrix(self):
        """Get reward matrix corresponding to the hexagon world.

        Reward matrix maps state and action to reward.
        Should be |states| x |actions|
        """
        states = [state for row in self.hexagons for state in row]
        num_states = len(states)
        R = np.zeros((num_states, 6))
        for row_index, row in enumerate(self.hexagons):
            for col_index, hexagon in enumerate(row):
                from_index = states.index(hexagon)
                for move in HexMove:
                    next_state = hexagon.get_next_hexagon(move)
                    if next_state == None:
                        R[from_index, move.value] = -1
                    if hexagon.score > 0:
                        R[from_index, move.value] = hexagon.score
        return R

    def get_mdp(self):
        return self.get_mdp_transition_matrix(), self.get_mdp_reward_matrix()

    def graph(self, ax, show_score=True, show_policy=False):
        vertices = np.array(
            [
                [0, 1],
                [np.sqrt(3) / 2, 0.5],
                [np.sqrt(3) / 2, -0.5],
                [0, -1],
                [-np.sqrt(3) / 2, -0.5],
                [-np.sqrt(3) / 2, 0.5],
                [0, 1],
            ]
        )
        for row_index, row in enumerate(self.hexagons):
            for col_index, hexagon in enumerate(row):
                verticies_new = vertices.copy()
                verticies_new[:, 0] += 1.75 * col_index
                verticies_new[:, 1] -= 1.5 * row_index
                if row_index % 2 != 0:
                    verticies_new[:, 0] += 1.75 / 2
                center = np.mean(verticies_new, axis=0)

                ax.plot(
                    verticies_new[:, 0],
                    verticies_new[:, 1],
                    "b-",
                )
                s = f"{hexagon.score}" if show_score else ""
                ax.text(
                    center[0],
                    center[1],
                    s,
                    ha="center",
                    va="center",
                )
                target_index = hexagon.policy.value - 1
                target_index_next = target_index + 1 if target_index + 1 <= 5 else 0
                target_vertex = (
                    0.5 * verticies_new[target_index]
                    + 0.5 * verticies_new[target_index_next]
                )
                if show_policy:
                    arrow_props = dict(arrowstyle="-|>", linewidth=2, color="red")
                    ax.annotate(
                        "",
                        xy=target_vertex,
                        xytext=center,
                        ha="center",
                        va="center",
                        arrowprops=arrow_props,
                    )

    def plt_graph(self, show_score=True, show_policy=False):
        fig, ax = plt.subplots(1, 1)
        self.graph(ax, show_score=show_score, show_policy=show_policy)
        plt.axis("equal")  # Set equal scaling for x and y axes
        plt.axis("off")


if __name__ == "__main__":
    grid = [["1", "2"], ["3", "4"]]
    hw = HexWorld(grid=grid)
    print(hw.hexagons[0][0].south_west)
