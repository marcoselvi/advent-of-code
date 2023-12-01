"""Tests for Advent of Code 2019 solutions."""

from google3.experimental.users.marcoselvi.advent import solve_2019
from google3.experimental.users.marcoselvi.advent import utils
from google3.testing.pybase import googletest


class AdventOfCode2019Test(googletest.TestCase):

  def test_day1(self):
    self.assertEqual((3427947, 5139037),
                     utils.runday(solve_2019.day1, '2019', 'day1'))

  def test_day2(self):
    self.assertEqual((6327510, 4112),
                     utils.runday(solve_2019.day2, '2019', 'day2'))

  def test_day3(self):
    self.assertEqual((870, 13698),
                     utils.runday(solve_2019.day3, '2019', 'day3'))

  def test_day4(self):
    self.assertEqual((481, 299),
                     utils.runday(solve_2019.day4, '2019', 'day4'))

  def test_day5(self):
    self.assertEqual((15508323, 9006327),
                     utils.runday(solve_2019.day5, '2019', 'day5'))

  def test_day6(self):
    self.assertEqual((402879, 484),
                     utils.runday(solve_2019.day6, '2019', 'day6'))

  def test_day7(self):
    self.assertEqual((22012, 4039164),
                     utils.runday(solve_2019.day7, '2019', 'day7'))

  def test_day8(self):
    self.assertEqual((1320,
                      '111   11  1   11  1 111  \n'
                      '1  1 1  1 1   11 1  1  1 \n'
                      '1  1 1     1 1 11   1  1 \n'
                      '111  1      1  1 1  111  \n'
                      '1 1  1  1   1  1 1  1 1  \n'
                      '1  1  11    1  1  1 1  1 '),
                     utils.runday(solve_2019.day8, '2019', 'day8'))

  def test_day9(self):
    self.assertEqual((2662308295, 63441),
                     utils.runday(solve_2019.day9, '2019', 'day9'))

  def test_day10(self):
    self.assertEqual((214, 502),
                     utils.runday(solve_2019.day10, '2019', 'day10'))

  def test_day11(self):
    self.assertEqual((),
                     utils.runday(solve_2019.day11, '2019', 'day11'))

  def test_day12(self):
    self.assertEqual((),
                     utils.runday(solve_2019.day12, '2019', 'day12'))

  def test_day13(self):
    self.assertEqual((),
                     utils.runday(solve_2019.day13, '2019', 'day13'))

  def test_day14(self):
    self.assertEqual((),
                     utils.runday(solve_2019.day14, '2019', 'day14'))

  def test_day15(self):
    self.assertEqual((),
                     utils.runday(solve_2019.day15, '2019', 'day15'))

  def test_day16(self):
    self.assertEqual((),
                     utils.runday(solve_2019.day16, '2019', 'day16'))

  def test_day17(self):
    self.assertEqual((),
                     utils.runday(solve_2019.day17, '2019', 'day17'))

  def test_day18(self):
    self.assertEqual((),
                     utils.runday(solve_2019.day18, '2019', 'day18'))

  def test_day19(self):
    self.assertEqual((),
                     utils.runday(solve_2019.day19, '2019', 'day19'))

  def test_day20(self):
    self.assertEqual((),
                     utils.runday(solve_2019.day20, '2019', 'day20'))

  def test_day21(self):
    self.assertEqual((),
                     utils.runday(solve_2019.day21, '2019', 'day21'))

  def test_day22(self):
    self.assertEqual((),
                     utils.runday(solve_2019.day22, '2019', 'day22'))

  def test_day23(self):
    self.assertEqual((),
                     utils.runday(solve_2019.day23, '2019', 'day23'))

  def test_day24(self):
    self.assertEqual((),
                     utils.runday(solve_2019.day24, '2019', 'day24'))

  def test_day25(self):
    self.assertEqual((),
                     utils.runday(solve_2019.day25, '2019', 'day25'))


if __name__ == '__main__':
  googletest.main()
