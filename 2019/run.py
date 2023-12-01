"""Executor of 2019 Advent of Code solutions."""
import sys

from . import solve
from . import utils


sys.setrecursionlimit(10000)


SELECTOR = {
    'day1': solve.day1,
    'day2': solve.day2,
    'day3': solve.day3,
    'day4': solve.day4,
    'day5': solve.day5,
    'day6': solve.day6,
    'day7': solve.day7,
    'day8': solve.day8,
    'day9': solve.day9,
    'day10': solve.day10,
    'day11': solve.day11,
    'day12': solve.day12,
    'day13': solve.day13,
    'day14': solve.day14,
    'day15': solve.day15,
    'day16': solve.day16,
    'day17': solve.day17,
    'day18': solve.day18,
    'day19': solve.day19,
    'day20': solve.day20,
    'day21': solve.day21,
    'day22': solve.day22,
    'day23': solve.day23,
    'day24': solve.day24,
    'day25': solve.day25,
}


def main(argv):
  print(utils.runday(SELECTOR[argv[1]], '2019', argv[1]))


if __name__ == '__main__':
  main(sys.argv)
