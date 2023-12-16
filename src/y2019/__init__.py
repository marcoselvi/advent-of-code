from . import solve

SELECTOR = [
    solve.day1,
    solve.day2,
    solve.day3,
    solve.day4,
    solve.day5,
    solve.day6,
    solve.day7,
    solve.day8,
    solve.day9,
    solve.day10,
    solve.day11,
    solve.day12,
    solve.day13,
    solve.day14,
    solve.day15,
    solve.day16,
    solve.day17,
    solve.day18,
    solve.day19,
    solve.day20,
    solve.day21,
    solve.day22,
    solve.day23,
    solve.day24,
    solve.day25,
]

def select(day):
  return SELECTOR[int(day) - 1]
