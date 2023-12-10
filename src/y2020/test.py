"""Tests for Advent of Code 2020 solutions."""

import sys

from . import solve
from .. import utils


sys.setrecursionlimit(10000)


def test_day1():
  assert (471019, 103927824) == utils.runday(solve.day1, '2020', 'day1')

def test_day2():
  assert (414, 413) == utils.runday(solve.day2, '2020', 'day2')

def test_day3():
  assert (220, 2138320800) == utils.runday(solve.day3, '2020', 'day3')

def test_day4():
  assert (235, 194) == utils.runday(solve.day4, '2020', 'day4')

def test_day5():
  assert (996, 671) == utils.runday(solve.day5, '2020', 'day5')

def test_day6():
  assert (7027, 3579) == utils.runday(solve.day6, '2020', 'day6')

def test_day7():
  assert (169, 82372) == utils.runday(solve.day7, '2020', 'day7')

def test_day8():
  assert (1331, 1121) == utils.runday(solve.day8, '2020', 'day8')

def test_day9():
  assert (530627549, 77730285) == utils.runday(solve.day9, '2020', 'day9')

def test_day10():
  assert (1848, 8099130339328) == utils.runday(solve.day10, '2020', 'day10')

def test_day11():
  assert (2194, 1944) == utils.runday(solve.day11, '2020', 'day11')

def test_day12():
  assert (938, 54404) == utils.runday(solve.day12, '2020', 'day12')

def test_day13():
  assert (2298, 783685719679632) == utils.runday(solve.day13, '2020', 'day13')

def test_day14():
  assert (7440382076205, 4200656704538) == utils.runday(solve.day14, '2020', 'day14')

def test_day15():
  assert (387, 6428) == utils.runday(solve.day15, '2020', 'day15')

def test_day16():
  assert (24110, 6766503490793) == utils.runday(solve.day16, '2020', 'day16')

def test_day17():
  assert (426, 1892) == utils.runday(solve.day17, '2020', 'day17')

def test_day18():
  assert (18213007238947, 388966573054664) == utils.runday(solve.day18, '2020', 'day18')

def test_day19():
  assert (118, 246) == utils.runday(solve.day19, '2020', 'day19')

def test_day20():
  assert (23386616781851, 2376) == utils.runday(solve.day20, '2020', 'day20')

def test_day21():
 assert (2485, 'bqkndvb zmb,bmrmhm,snhrpv,vflms,bqtvr,qzkjrtl,rkkrx') == utils.runday(solve.day21, '2020', 'day21')

def test_day22():
  assert (33010, 32769) == utils.runday(solve.day22, '2020', 'day22')

def test_day23():
  assert (47382659, 42271866720) == utils.runday(solve.day23, '2020', 'day23')

def test_day24():
  assert (427, 3837) == utils.runday(solve.day24, '2020', 'day24')

def test_day25():
  assert 4441893 == utils.runday(solve.day25  '2020', 'day25')
