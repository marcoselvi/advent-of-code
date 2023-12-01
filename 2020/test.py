"""Tests for Advent of Code 2020 solutions."""

import sys

from google3.experimental.users.marcoselvi.advent import solve_2020
from google3.experimental.users.marcoselvi.advent import utils
from google3.testing.pybase import googletest

sys.setrecursionlimit(10000)


class AdventOfCode2020Test(googletest.TestCase):

  def test_day1(self):
    self.assertEqual((471019, 103927824),
                     utils.runday(solve_2020.day1, '2020', 'day1'))

  def test_day2(self):
    self.assertEqual((414, 413),
                     utils.runday(solve_2020.day2, '2020', 'day2'))

  def test_day3(self):
    self.assertEqual((220, 2138320800),
                     utils.runday(solve_2020.day3, '2020', 'day3'))

  def test_day4(self):
    self.assertEqual((235, 194),
                     utils.runday(solve_2020.day4, '2020', 'day4'))

  def test_day5(self):
    self.assertEqual((996, 671),
                     utils.runday(solve_2020.day5, '2020', 'day5'))

  def test_day6(self):
    self.assertEqual((7027, 3579),
                     utils.runday(solve_2020.day6, '2020', 'day6'))

  def test_day7(self):
    self.assertEqual((169, 82372),
                     utils.runday(solve_2020.day7, '2020', 'day7'))

  def test_day8(self):
    self.assertEqual((1331, 1121),
                     utils.runday(solve_2020.day8, '2020', 'day8'))

  def test_day9(self):
    self.assertEqual((530627549, 77730285),
                     utils.runday(solve_2020.day9, '2020', 'day9'))

  def test_day10(self):
    self.assertEqual((1848, 8099130339328),
                     utils.runday(solve_2020.day10, '2020', 'day10'))

  def test_day11(self):
    self.assertEqual((2194, 1944),
                     utils.runday(solve_2020.day11, '2020', 'day11'))

  def test_day12(self):
    self.assertEqual((938, 54404),
                     utils.runday(solve_2020.day12, '2020', 'day12'))

  def test_day13(self):
    self.assertEqual((2298, 783685719679632),
                     utils.runday(solve_2020.day13, '2020', 'day13'))

  def test_day14(self):
    self.assertEqual((7440382076205, 4200656704538),
                     utils.runday(solve_2020.day14, '2020', 'day14'))

  def test_day15(self):
    self.assertEqual((387, 6428),
                     utils.runday(solve_2020.day15, '2020', 'day15'))

  def test_day16(self):
    self.assertEqual((24110, 6766503490793),
                     utils.runday(solve_2020.day16, '2020', 'day16'))

  def test_day17(self):
    self.assertEqual((426, 1892),
                     utils.runday(solve_2020.day17, '2020', 'day17'))

  def test_day18(self):
    self.assertEqual((18213007238947, 388966573054664),
                     utils.runday(solve_2020.day18, '2020', 'day18'))

  def test_day19(self):
    self.assertEqual((118, 246),
                     utils.runday(solve_2020.day19, '2020', 'day19'))

  def test_day20(self):
    self.assertEqual((23386616781851, 2376),
                     utils.runday(solve_2020.day20, '2020', 'day20'))

  def test_day21(self):
    self.assertEqual((2485,
                      'bqkndvb,zmb,bmrmhm,snhrpv,vflms,bqtvr,qzkjrtl,rkkrx'),
                     utils.runday(solve_2020.day21, '2020', 'day21'))

  def test_day22(self):
    self.assertEqual((33010, 32769),
                     utils.runday(solve_2020.day22, '2020', 'day22'))

  def test_day23(self):
    self.assertEqual((47382659, 42271866720),
                     utils.runday(solve_2020.day23, '2020', 'day23'))

  def test_day24(self):
    self.assertEqual((427, 3837),
                     utils.runday(solve_2020.day24, '2020', 'day24'))

  def test_day25(self):
    self.assertEqual(4441893,
                     utils.runday(solve_2020.day25, '2020', 'day25'))


if __name__ == '__main__':
  googletest.main()
