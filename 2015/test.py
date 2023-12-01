"""Tests for Advent of Code 2015 solutions."""

from google3.experimental.users.marcoselvi.advent import solve_2015
from google3.experimental.users.marcoselvi.advent import utils
from google3.testing.pybase import googletest


class AdventOfCode2015Test(googletest.TestCase):

  def test_day1(self):
    self.assertEqual((280, 1797),
                     utils.runday(solve_2015.day1, '2015', 'day1'))

  def test_day2(self):
    self.assertEqual((1586300, 3737498),
                     utils.runday(solve_2015.day2, '2015', 'day2'))

  def test_day3(self):
    self.assertEqual((2572, 2631),
                     utils.runday(solve_2015.day3, '2015', 'day3'))

  def test_day4(self):
    self.assertEqual((254575, 1038736),
                     utils.runday(solve_2015.day4, '2015', 'day4'))

  def test_day5(self):
    self.assertEqual((238, 69),
                     utils.runday(solve_2015.day5, '2015', 'day5'))

  def test_day6(self):
    self.assertEqual((377891, 14110788),
                     utils.runday(solve_2015.day6, '2015', 'day6'))

  def test_day7(self):
    self.assertEqual((3176, 14710),
                     utils.runday(solve_2015.day7, '2015', 'day7'))

  def test_day8(self):
    self.assertEqual((1371, 2117),
                     utils.runday(solve_2015.day8, '2015', 'day8'))

  def test_day9(self):
    self.assertEqual((207, 804),
                     utils.runday(solve_2015.day9, '2015', 'day9'))

  def test_day10(self):
    self.assertEqual((360154, 5103798),
                     utils.runday(solve_2015.day10, '2015', 'day10'))

  def test_day11(self):
    self.assertEqual(('hepxxyzz', 'heqaabcc'),
                     utils.runday(solve_2015.day11, '2015', 'day11'))

  def test_day12(self):
    self.assertEqual((119433, 68466),
                     utils.runday(solve_2015.day12, '2015', 'day12'))

  def test_day13(self):
    self.assertEqual((),
                     utils.runday(solve_2015.day13, '2015', 'day13'))

  def test_day14(self):
    self.assertEqual((),
                     utils.runday(solve_2015.day14, '2015', 'day14'))

  def test_day15(self):
    self.assertEqual((),
                     utils.runday(solve_2015.day15, '2015', 'day15'))

  def test_day16(self):
    self.assertEqual((),
                     utils.runday(solve_2015.day16, '2015', 'day16'))

  def test_day17(self):
    self.assertEqual((),
                     utils.runday(solve_2015.day17, '2015', 'day17'))

  def test_day18(self):
    self.assertEqual((),
                     utils.runday(solve_2015.day18, '2015', 'day18'))

  def test_day19(self):
    self.assertEqual((),
                     utils.runday(solve_2015.day19, '2015', 'day19'))

  def test_day20(self):
    self.assertEqual((),
                     utils.runday(solve_2015.day20, '2015', 'day20'))

  def test_day21(self):
    self.assertEqual((),
                     utils.runday(solve_2015.day21, '2015', 'day21'))

  def test_day22(self):
    self.assertEqual((),
                     utils.runday(solve_2015.day22, '2015', 'day22'))

  def test_day23(self):
    self.assertEqual((),
                     utils.runday(solve_2015.day23, '2015', 'day23'))

  def test_day24(self):
    self.assertEqual((),
                     utils.runday(solve_2015.day24, '2015', 'day24'))

  def test_day25(self):
    self.assertEqual((),
                     utils.runday(solve_2015.day25, '2015', 'day25'))


if __name__ == '__main__':
  googletest.main()
