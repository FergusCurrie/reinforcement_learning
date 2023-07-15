import pytest
import numpy as np
from hex_world import HexWorld


def test_small_hex_world():
    grid = [["1", "2"], ["3", "4"]]
    hw = HexWorld(grid=grid)
    # check top left hexagon
    assert hw.hexagons[0][0].west == None
    assert hw.hexagons[0][0].north_east == None
    assert hw.hexagons[0][0].north_west == None
    assert hw.hexagons[0][0].south_west == None
    assert hw.hexagons[0][0].south_east.score == 3
    assert hw.hexagons[0][0].east.score == 2

    # check top right hexagon
    assert hw.hexagons[0][1].west.score == 1
    assert hw.hexagons[0][1].north_east == None
    assert hw.hexagons[0][1].north_west == None
    assert hw.hexagons[0][1].south_west.score == 3
    assert hw.hexagons[0][1].south_east.score == 4
    assert hw.hexagons[0][1].east == None

    # check bot right hexagon
    assert hw.hexagons[1][1].west.score == 3
    assert hw.hexagons[1][1].north_east == None
    assert hw.hexagons[1][1].north_west.score == 2
    assert hw.hexagons[1][1].south_west == None
    assert hw.hexagons[1][1].south_east == None
    assert hw.hexagons[1][1].east == None

    # check bot left hexagon
    assert hw.hexagons[1][0].west == None
    assert hw.hexagons[1][0].north_east.score == 2
    assert hw.hexagons[1][0].north_west.score == 1
    assert hw.hexagons[1][0].south_west == None
    assert hw.hexagons[1][0].south_east == None
    assert hw.hexagons[1][0].east.score == 4
