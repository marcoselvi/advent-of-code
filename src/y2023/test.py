"""Tests for Advent of Code 2023 solutions."""

import sys

from . import solve
from .. import utils


sys.setrecursionlimit(100000)


def test_day1():
  assert utils.runday(solve.day1, '2023', 'day1') == (56506, 56017)


def test_day2():
  assert utils.runday(solve.day2, '2023', 'day2') == (1931, 83105)


def test_day3():
  assert utils.runday(solve.day3, '2023', 'day3') == (544433, 76314915)


def test_day4():
  assert utils.runday(solve.day4, '2023', 'day4') == (25010, 9924412)


def test_day5():
  assert utils.runday(solve.day5, '2023', 'day5') == (462648396, 2520479)


def test_day6():
  assert utils.runday(solve.day6, '2023', 'day6') == (170000, 20537782)


def test_day7():
  assert utils.runday(solve.day7, '2023', 'day7') == (247815719, 248747492)


def test_day8():
  assert utils.runday(solve.day8, '2023', 'day8') == (16531, 24035773251517)


def test_day9():
  assert utils.runday(solve.day9, '2023', 'day9') == (1834108701, 993)


def test_day10():
  assert utils.runday(solve.day10, '2023', 'day10') == (6786, 495)


def test_day11():
  assert utils.runday(solve.day11, '2023', 'day11') == (9509330, 635832237682)


def test_day12():
  assert utils.runday(solve.day12, '2023', 'day12') == (7843, 10153896718999)


def test_day13():
  assert utils.runday(solve.day13, '2023', 'day13') == (41859, 30842)


def test_day14():
  assert utils.runday(solve.day14, '2023', 'day14') == (106517, 79723)


def test_day15():
  assert utils.runday(solve.day15, '2023', 'day15') == (510792, 269410)


def test_day16():
  assert utils.runday(solve.day16, '2023', 'day16') == (7242, 7572)


def test_day17():
  assert utils.runday(solve.day17, '2023', 'day17') == ()


def test_day18():
  assert utils.runday(solve.day18, '2023', 'day18') == ()


def test_day19():
  assert utils.runday(solve.day19, '2023', 'day19') == ()


def test_day20():
  assert utils.runday(solve.day20, '2023', 'day20') == ()


def test_day21():
  assert utils.runday(solve.day21, '2023', 'day21') == ()


def test_day22():
  assert utils.runday(solve.day22, '2023', 'day22') == ()


def test_day23():
  assert utils.runday(solve.day23, '2023', 'day23') == ()


def test_day24():
  assert utils.runday(solve.day24, '2023', 'day24') == ()


def test_day25():
  assert utils.runday(solve.day25, '2023', 'day25') == ()
