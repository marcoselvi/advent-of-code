import re

import functools as fnt


WORD_TO_DIGIT_MAP = {
    'zero': '0',
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9',
}


def get_lines(path):
  with open(path, 'r') as f:
    return f.readlines()


def day1_p1():
  def digits(line):
    numbers = re.findall(r'\d', line)
    return int(numbers[0] + numbers[-1])
  def add_up(lines):
    return sum(digits(line) for line in lines)
  testlines = get_lines('src/2023/day1.calibration.test1.txt')
  lines = get_lines('src/2023/day1.calibration.txt')
  return add_up(testlines), add_up(lines)

def day1_p2():
  def digits(line):
    pattern = '|'.join(WORD_TO_DIGIT_MAP.keys())
    numbers = [WORD_TO_DIGIT_MAP.get(maybe_word[0], maybe_word[1])
               for maybe_word in re.findall(rf'(?=({pattern}))|(\d)', line)]
    return int(numbers[0] + numbers[-1])
  def add_up(lines):
    return sum(digits(line) for line in lines)
  testlines = get_lines('src/2023/day1.calibration.test2.txt')
  lines = get_lines('src/2023/day1.calibration.txt')
  return add_up(testlines), add_up(lines)


def game_rounds(game):
  return [[p.strip().split(' ') for p in r.split(',')] for r in game.split(':')[-1].split(';')]

def day2_p1():
  available = {'red': 12, 'green': 13, 'blue': 14}
  def is_possible(game):
    for r in game_rounds(game):
      for n, c in r:
        if available[c] < int(n):
          return False
    return True
  def possible_games(games):
    return filter(is_possible, games)
  def game_id(game):
    return int(game.split(':')[0].split(' ')[-1])
  def add_up(games):
    return sum(game_id(game) for game in games)
  testgames = get_lines('src/2023/day2.cubes.test1.txt')
  games = get_lines('src/2023/day2.cubes.txt')
  return add_up(possible_games(testgames)), add_up(possible_games(games))

def day2_p2():
  def power(game):
    rounds = game_rounds(game)
    reds = [int(n) for r in rounds for n, c in r if c == 'red']
    blues = [int(n) for r in rounds for n, c in r if c == 'blue']
    greens = [int(n) for r in rounds for n, c in r if c == 'green']
    return max(reds) * max(blues) * max(greens)
  testgames = get_lines('src/2023/day2.cubes.test1.txt')
  games = get_lines('src/2023/day2.cubes.txt')
  return sum(power(game) for game in testgames), sum(power(game) for game in games)


def day3_p1():
  def near(xs_y, sx_sy):
    (xs, y), (sx, sy) = xs_y, sx_sy
    return (min(xs)-1 <= sx <= max(xs)+1) and (y-1 <= sy <= y+1)
  def match(numbers, symbols):
    return [n for (n, (xs, y)) in numbers
            if any(near((xs, y), (sx, sy)) for (_, (sx, sy)) in symbols)]
  def row_elements(row):
    def accum_elements(elem_track, i_ch):
      (elements, (digits, xs)), (i, ch) = elem_track, i_ch
      return ((elements, (digits+ch, xs+(i,))) if ch.isdigit() else
              (elements+[(digits, xs), (ch, i)], ('', ())) if ch != '.' and digits else
              (elements+[(ch, i)], ('', ())) if ch != '.' else
              (elements+[(digits, xs)], ('', ())) if digits else
              (elements, ('', ())))
    return fnt.reduce(accum_elements, enumerate(row + '.'), ([], ('', ())))[0]
  def numbers_and_symbols(rows):
    elements = [(elem, (x, j)) for j, row in enumerate(rows) for elem, x in row_elements(row.strip())]
    return ([(int(elem), x_y) for elem, x_y in elements if elem.isdigit()],
            [(elem, x_y) for elem, x_y in elements if not elem.isdigit()])
  testrows = get_lines('src/2023/day3.engine.test1.txt')
  rows = get_lines('src/2023/day3.engine.txt')
  # schema: (number, ((xs,), y)), (symbol, (x, y))
  return sum(match(*numbers_and_symbols(testrows))), sum(match(*numbers_and_symbols(rows)))

def day3_p2():
  def near(xs_y, sx_sy):
    (xs, y), (sx, sy) = xs_y, sx_sy
    return (min(xs)-1 <= sx <= max(xs)+1) and (y-1 <= sy <= y+1)
  def match(numbers, maybe_gears):
    gears = []
    for _, (gx, gy) in maybe_gears:
      near_numbers = tuple(filter(lambda n_xs_y: near(n_xs_y[1], (gx, gy)), numbers))
      if len(near_numbers) == 2:
        gears.append((near_numbers[0][0], near_numbers[1][0]))
    return gears
  def row_elements(row):
    def accum_elements(elem_track, i_ch):
      (elements, (digits, xs)), (i, ch) = elem_track, i_ch
      return ((elements, (digits+ch, xs+(i,))) if ch.isdigit() else
              (elements+[(digits, xs), (ch, i)], ('', ())) if ch == '*' and digits else
              (elements+[(ch, i)], ('', ())) if ch == '*' else
              (elements+[(digits, xs)], ('', ())) if digits else
              (elements, ('', ())))
    return fnt.reduce(accum_elements, enumerate(row + '.'), ([], ('', ())))[0]
  def numbers_and_gears(rows):
    elements = [(elem, (x, j)) for j, row in enumerate(rows) for elem, x in row_elements(row.strip())]
    return ([(int(elem), x_y) for elem, x_y in elements if elem.isdigit()],
            [(elem, x_y) for elem, x_y in elements if not elem.isdigit()])
  def gear_ratios(gears):
    return sum(n1*n2 for n1, n2 in gears)
  testrows = get_lines('src/2023/day3.engine.test1.txt')
  rows = get_lines('src/2023/day3.engine.txt')
  # schema: (number, ((xs,), y)), (symbol, (x, y))
  return (gear_ratios(match(*numbers_and_gears(testrows))),
          gear_ratios(match(*numbers_and_gears(rows))))


def day4_p1():
  def points(row):
    def split(s):
      return filter(None, s.strip().split(' '))
    _, numbers = row.split(':')
    winning, got = numbers.strip().split('|')
    intersection = set(split(winning)) & set(split(got))
    return int(2**(len(intersection) - 1)) if intersection else 0
  testrows = get_lines('src/2023/day4.scratchcards.test1.txt')
  rows = get_lines('src/2023/day4.scratchcards.txt')
  return sum(points(row) for row in testrows), sum(points(row) for row in rows)

def day4_p2():
  def card_id(row):
    return int(row.split(':')[0].split(' ')[-1].strip())
  def winning_ns(row):
    return set(filter(None, row.split(':')[-1].split('|')[0].strip().split(' ')))
  def got_ns(row):
    return set(filter(None, row.split(':')[-1].split('|')[-1].strip().split(' ')))
  def n_cards(rows):
    cards = [(card_id(row), winning_ns(row), got_ns(row)) for row in rows]
    counter = {cid: 1 for cid, _, _ in cards}
    for cid, win_n, got_n in cards:
      inter = win_n & got_n
      new_cards = cards[cid:(cid+len(inter))]
      counter = counter | {ncid: (counter[ncid] + counter[cid]) for ncid, _, _ in new_cards}
    return sum(counter.values())
  testrows = get_lines('src/2023/day4.scratchcards.test1.txt')
  rows = get_lines('src/2023/day4.scratchcards.txt')
  return n_cards(testrows), n_cards(rows)


if __name__ == '__main__':
  print('Day 1 solutions:', day1_p1(), day1_p2())
  print('Day 2 solutions:', day2_p1(), day2_p2())
  print('Day 3 solutions:', day3_p1(), day3_p2())
  print('Day 4 solutions:', day4_p1(), day4_p2())
