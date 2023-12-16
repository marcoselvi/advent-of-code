"""Tests for Advent of Code 2020 solutions."""

import sys

from . import solve
from .. import utils


sys.setrecursionlimit(10000)


def test_day1():
  assert utils.runday(solve.day1, '2020', 'day1') == (471019, 103927824)

def test_day2():
  assert utils.runday(solve.day2, '2020', 'day2') == (414, 413)

def test_day3():
  assert utils.runday(solve.day3, '2020', 'day3') == (220, 2138320800)

def test_day4():
  assert utils.runday(solve.day4, '2020', 'day4') == (235, 194)

def test_day5():
  assert utils.runday(solve.day5, '2020', 'day5') == (996, 671)

def test_day6():
  assert utils.runday(solve.day6, '2020', 'day6') == (7027, 3579)

def test_day7():
  assert utils.runday(solve.day7, '2020', 'day7') == (169, 82372)

def test_day8():
  assert utils.runday(solve.day8, '2020', 'day8') == (1331, 1121)

def test_day9():
  assert utils.runday(solve.day9, '2020', 'day9') == (530627549, 77730285)

def test_day10():
  assert utils.runday(solve.day10, '2020', 'day10') == (1848, 8099130339328)

def test_day11():
  assert utils.runday(solve.day11, '2020', 'day11') == (2194, 1944)

def test_day12():
  assert utils.runday(solve.day12, '2020', 'day12') == (938, 54404)

def test_day13():
  assert utils.runday(solve.day13, '2020', 'day13') == (2298, 783685719679632)

def test_day14():
  assert utils.runday(solve.day14, '2020', 'day14') == (7440382076205, 4200656704538)

def test_day15():
  assert utils.runday(solve.day15, '2020', 'day15') == (387, 6428)

def test_day16():
  assert utils.runday(solve.day16, '2020', 'day16') == (24110, 6766503490793)

def test_day17():
  assert utils.runday(solve.day17, '2020', 'day17') == (426, 1892)

def test_day18():
  assert utils.runday(solve.day18, '2020', 'day18') == (18213007238947, 388966573054664)

def test_day19():
  assert utils.runday(solve.day19, '2020', 'day19') == (118, 246)

def test_day20():
  assert utils.runday(solve.day20, '2020', 'day20') == (23386616781851, 2376)

def test_day21():
 assert utils.runday(solve.day21, '2020', 'day21') == (2485, 'bqkndvb,zmb,bmrmhm,snhrpv,vflms,bqtvr,qzkjrtl,rkkrx')

def test_day22():
  assert utils.runday(solve.day22, '2020', 'day22') == (33010, 32769)

def test_day23():
  assert utils.runday(solve.day23, '2020', 'day23') == (47382659, 42271866720)

def test_day24():
  assert utils.runday(solve.day24, '2020', 'day24') == (427, 3837)

def test_day25():
  assert utils.runday(solve.day25, '2020', 'day25') == 4441893
