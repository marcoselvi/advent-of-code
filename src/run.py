"""Executor of Advent of Code solutions."""
import sys

from . import utils as ut
from . import y2015
from . import y2019
from . import y2020
from . import y2023


sys.setrecursionlimit(100000)


YEAR_SELECTOR = {
    '2015': y2015,
    '2019': y2019,
    '2020': y2020,
    '2023': y2023,
}


def main(argv):
  year, day = argv[1:]
  print(ut.runday(getattr(YEAR_SELECTOR[year].solve, f'day{day}'), year, f'day{day}'))


if __name__ == '__main__':
  main(sys.argv)
