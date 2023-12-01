"""Executor of 2019 Advent of Code solutions."""
import sys

from absl import app

from google3.experimental.users.marcoselvi.advent import solve_2019
from google3.experimental.users.marcoselvi.advent import utils


sys.setrecursionlimit(10000)


SELECTOR = {
    'day1': solve_2019.day1,
    'day2': solve_2019.day2,
    'day3': solve_2019.day3,
    'day4': solve_2019.day4,
    'day5': solve_2019.day5,
    'day6': solve_2019.day6,
    'day7': solve_2019.day7,
    'day8': solve_2019.day8,
    'day9': solve_2019.day9,
    'day10': solve_2019.day10,
    'day11': solve_2019.day11,
    'day12': solve_2019.day12,
    'day13': solve_2019.day13,
    'day14': solve_2019.day14,
    'day15': solve_2019.day15,
    'day16': solve_2019.day16,
    'day17': solve_2019.day17,
    'day18': solve_2019.day18,
    'day19': solve_2019.day19,
    'day20': solve_2019.day20,
    'day21': solve_2019.day21,
    'day22': solve_2019.day22,
    'day23': solve_2019.day23,
    'day24': solve_2019.day24,
    'day25': solve_2019.day25,
}


def main(argv):
  print(utils.runday(SELECTOR[argv[1]], '2019', argv[1]))


if __name__ == '__main__':
  app.run(main)

