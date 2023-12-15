from collections import Counter
import math
import re
import sys

import functools as fnt
import itertools as it

sys.setrecursionlimit(100000)


def get_rows(path):
  with open(path, 'r') as f:
    return list(f.readlines())


def fst(x_y): return x_y[0]


def join(xss):
  return [x for xs in xss for x in xs]


def memoise(f):
  memo = {}
  def g(*x):
    if x not in memo:
      memo[x] = f(*x)
    return memo[x]
  return g


@memoise
def transpose(lines):
  return tuple(''.join(c) for c in zip(*lines))


def raise_(e):
  raise ValueError(e)


def day1():

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

  def add_up(digits, rows):
    return sum(digits(row) for row in rows)

  rows = get_rows('data/y2023/day1.calibration.txt')

  def p1():
    def digits(row):
      numbers = re.findall(r'\d', row)
      return int(numbers[0] + numbers[-1])
    return add_up(digits, rows)

  def p2():
    def digits(row):
      pattern = '|'.join(WORD_TO_DIGIT_MAP.keys())
      numbers = [WORD_TO_DIGIT_MAP.get(maybe_word[0], maybe_word[1])
                 for maybe_word in re.findall(rf'(?=({pattern}))|(\d)', row)]
      return int(numbers[0] + numbers[-1])
    return add_up(digits, rows)

  return p1(), p2()


def day2():

  def game_rounds(game):
    return [[p.strip().split(' ') for p in r.split(',')] for r in game.split(':')[-1].split(';')]

  games = get_rows('data/y2023/day2.cubes.txt')

  def p1():
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
    return sum(game_id(game) for game in possible_games(games))

  def p2():
    def power(game):
      rounds = game_rounds(game)
      reds = [int(n) for r in rounds for n, c in r if c == 'red']
      blues = [int(n) for r in rounds for n, c in r if c == 'blue']
      greens = [int(n) for r in rounds for n, c in r if c == 'green']
      return max(reds) * max(blues) * max(greens)
    return sum(power(game) for game in games)

  return p1(), p2()


def day3():

  def near(xs_y, sx_sy):
    (xs, y), (sx, sy) = xs_y, sx_sy
    return (min(xs)-1 <= sx <= max(xs)+1) and (y-1 <= sy <= y+1)

  rows = get_rows('data/y2023/day3.engine.txt')

  def p1():
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
    rows = get_rows('data/y2023/day3.engine.txt')
    # schema: (number, ((xs,), y)), (symbol, (x, y))
    return sum(match(*numbers_and_symbols(rows)))

  def p2():
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
    # schema: (number, ((xs,), y)), (symbol, (x, y))
    return gear_ratios(match(*numbers_and_gears(rows)))

  return p1(), p2()


def day4():

  rows = get_rows('data/y2023/day4.scratchcards.txt')

  def p1():
    def points(row):
      def split(s):
        return filter(None, s.strip().split(' '))
      _, numbers = row.split(':')
      winning, got = numbers.strip().split('|')
      intersection = set(split(winning)) & set(split(got))
      return int(2**(len(intersection) - 1)) if intersection else 0
    return sum(points(row) for row in rows)

  def p2():
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
    return n_cards(rows)

  return p1(), p2()


def day5():

  rows = get_rows('data/y2023/day5.seedmaps.txt')

  def p1():
    def map_name(map_str):
      return map_str.split(':')[0].split(' ')[0].strip()
    def map_fn(map_str):
      range_values = list(filter(None, map_str.split('\n')[1:]))
      def fn(x):
        for mv, mk, s in [map(int, ran.strip().split(' ')) for ran in range_values]:
          if mk <= x < mk+s:
            return mv + x - mk
        return x
      return fn
    def lowest_location(rows):
      seeds = [int(seed.strip()) for seed in rows[0].split(':')[1].strip().split(' ')]
      maps_str = ''.join(rows[2:])
      maps = [(map_name(map_str), map_fn(map_str.strip())) for map_str in maps_str.split('\n\n')]
      return min(fnt.reduce(lambda s, m: m[1](s), maps, seed) for seed in seeds)
    return lowest_location(rows)

  def p2():
    def map_ranges(map_str):
      range_values = list(filter(None, map_str.split('\n')[1:]))
      return [list(map(int, ran.strip().split(' '))) for ran in range_values]
    def bisect_range(r, m_maps):
      rk, rs = r
      for mv, mk, ms in m_maps:
        if mk <= rk < mk + ms:
          new_rk = mv + rk - mk
          if rk + rs >= mk + ms:
            return [(new_rk, mk + ms - rk)] + bisect_range((mk + ms, rs - mk - ms + rk), m_maps)
          return [(new_rk, rs)]
      return [(rk, rs)]
    def ranges_to_ranges(seed_ranges, map_ranges):
      return [new_range for r in seed_ranges for new_range in bisect_range(r, map_ranges)]
    def lowest_location(rows):
      seed_values = list(map(int, rows[0].split(':')[1].strip().split(' ')))
      seed_ranges = list(zip(seed_values[::2], seed_values[1::2]))
      maps_str = ''.join(rows[2:])
      maps = list(map(map_ranges, maps_str.split('\n\n')))
      return min(map(fst, fnt.reduce(ranges_to_ranges, maps, seed_ranges)))
    return lowest_location(rows)

  return p1(), p2()


def day6():

  rows = get_rows('data/y2023/day6.boatsboatsboats.txt')

  def p1():
    def process_data(row):
      return [int(n.strip()) for n in row.split(':')[1].strip().split(' ') if n]
    def scores(time):
      return [n * (time - n) for n in range(time)]
    def better_times(rows):
      races = list(zip(*map(process_data, rows)))
      all_scores = [(best, scores(time)) for time, best in races]
      return fnt.reduce(lambda x, y: x*y, (len([s for s in ss if s > best]) for best, ss in all_scores), 1)
    return better_times(rows)

  def p2():
    def process_data(row):
      numbers = [n.strip() for n in row.split(':')[1].strip().split(' ')]
      return int(''.join(numbers))
    def scores(time):
      return [n * (time - n) for n in range(time)]
    def better_times(rows):
      time, best = process_data(rows[0]), process_data(rows[1])
      return len([s for s in scores(time) if s > best])
    return better_times(rows)

  return p1(), p2()


def day7():

  rows = get_rows('data/y2023/day7.poker.txt')

  def p1():
    def card_values(h):
      return tuple(int({'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10}.get(c, c))
                   for c in h)
    def hand_value(h):
      h_c = Counter(h)
      max_c = max(h_c.values())
      return (6 if max_c == 5 else
              5 if max_c == 4 else
              4 if len(h_c) == 2 and max_c == 3 else
              3 if max_c == 3 else
              2 if len(h_c) == 3 and max_c == 2 else
              1 if max_c == 2 else
              0)
    def hand_sorting(hand_bet):
      hand, _ = hand_bet
      return (hand_value(hand), card_values(hand))
    def winnings(rows):
      return sum((i+1)*int(bet) for i, (hand, bet)
                 in enumerate(sorted([row.strip().split(' ') for row in rows],
                                     key=hand_sorting)))
    return winnings(rows)

  def p2():
    def card_values(h):
      return tuple(int({'A': 14, 'K': 13, 'Q': 12, 'J': 1, 'T': 10}.get(c, c))
                   for c in h)
    def hand_value(h):
      js = Counter(h).get('J', 0)
      h_c = Counter(c for c in h if c != 'J')
      max_c = max(list(h_c.values()) or [0])
      return (6 if max_c + js == 5 else
              5 if max_c + js == 4 else
              4 if len(h_c) == 2 and max_c + js == 3 else
              3 if max_c + js == 3 else
              2 if len(h_c) == 3 and max_c + js == 2 else
              1 if max_c + js == 2 else
              0)
    def hand_sorting(hand_bet):
      hand, _ = hand_bet
      return (hand_value(hand), card_values(hand))
    def winnings(rows):
      return sum((i+1)*int(bet) for i, (hand, bet)
                 in enumerate(sorted([row.strip().split(' ') for row in rows],
                                     key=hand_sorting)))
    return winnings(rows)

  return p1(), p2()


def day8():

  rows = get_rows('data/y2023/day8.maps.txt')

  def p1():
    def traverse(rows):
      instructions = rows[0].strip()
      graph = {node.strip(): tuple(left_right.strip().strip('(').strip(')').split(', '))
               for node, left_right in [row.split(' = ') for row in rows[2:]]}
      steps = 0
      node = 'AAA'
      for lr in it.cycle(instructions):
        node = graph[node][0] if lr == 'L' else graph[node][1]
        steps += 1
        if node == 'ZZZ':
          return steps
    return traverse(rows)

  def p2():
    def traverse(rows):
      instructions = rows[0].strip()
      graph = {node.strip(): tuple(left_right.strip().strip('(').strip(')').split(', '))
               for node, left_right in [row.split(' = ') for row in rows[2:]]}

      def unroll(node):
        steps = 0
        for lr in it.cycle(instructions):
          node = graph[node][0] if lr == 'L' else graph[node][1]
          steps += 1
          if node.endswith('Z'):
            return steps

      nodes = [node for node in graph.keys() if node.endswith('A')]
      return math.lcm(*(unroll(node) for node in graph.keys() if node.endswith('A')))

    return traverse(rows)

  return p1(), p2()


def day9():

  def parse(rows):
    return [list(map(int, row.strip().split(' '))) for row in rows]

  parsed_rows = parse(get_rows('data/y2023/day9.oasis.txt'))

  def p1():
    def subseq(seq):
      return [(b - a) for a, b in zip(seq, seq[1:])]
    def next_value(seq):
      return seq[-1] + next_value(subseq(seq)) if not all(e == 0 for e in seq) else 0
    def sum_next_values(rows):
      return sum(next_value(row) for row in rows)
    return sum_next_values(parsed_rows)

  def p2():
    def subseq(seq):
      return [(b - a) for a, b in zip(seq, seq[1:])]
    def prev_value(seq):
      return seq[0] - prev_value(subseq(seq)) if not all(e == 0 for e in seq) else 0
    def sum_prev_values(rows):
      return sum(prev_value(row) for row in rows)
    return sum_prev_values(parsed_rows)

  return p1(), p2()


def day10():

  def north(x_y):
    x, y = x_y
    return (x, y-1)
  def south(x_y):
    x, y = x_y
    return (x, y+1)
  def east(x_y):
    x, y = x_y
    return (x+1, y)
  def west(x_y):
    x, y = x_y
    return (x-1, y)

  def parse(rows):
    return {(x, y): pipe for y, row in enumerate(rows) for x, pipe in enumerate(row)}

  pipes = parse(get_rows('data/y2023/day10.pipes.txt'))

  def connect(x_y):
    p = pipes[x_y]
    return ((north(x_y), south(x_y)) if p == '|' else
            (east(x_y), west(x_y)) if p == '-' else
            (north(x_y), east(x_y)) if p == 'L' else
            (north(x_y), west(x_y)) if p == 'J' else
            (south(x_y), west(x_y)) if p == '7' else
            (south(x_y), east(x_y)) if p == 'F' else
            raise_(f'invalid connection {x_y}, {pipe}'))

  def rconnect(x_y):
    n, s, w, e = north(x_y), south(x_y), west(x_y), east(x_y)
    return (  ((n,) if pipes.get(n, '') in {'|', '7', 'F'} else ((),))
            + ((s,) if pipes.get(s, '') in {'|', 'L', 'J'} else ((),))
            + ((w,) if pipes.get(w, '') in {'-', 'L', 'F'} else ((),))
            + ((e,) if pipes.get(e, '') in {'-', '7', 'J'} else ((),)))

  def find_S(pipes):
    for x_y, p in pipes.items():
      if p == 'S':
        return x_y

  sx_sy = find_S(pipes)

  def convert_S(sx_sy):
    booleans = list(map(bool, rconnect(sx_sy)))
    return pipes | {sx_sy: ('|' if booleans == [1, 1, 0, 0] else
                            '-' if booleans == [0, 0, 1, 1] else
                            'L' if booleans == [1, 0, 0, 1] else
                            'J' if booleans == [1, 0, 1, 0] else
                            '7' if booleans == [0, 1, 1, 0] else
                            'F' if booleans == [0, 1, 0, 1] else
                            raise_(f'invalid S {sx_sy}, {booleans}'))}

  def step(visited, px_py, x_y, s):
    if x_y in visited:
      return visited, s
    next_xy = (set(connect(x_y)) - set((px_py,))).pop()
    return step(visited | set((x_y,)), x_y, next_xy, s+1)

  start, _ = [p for p in rconnect(sx_sy) if p]
  visited, steps = step(set((sx_sy,)), sx_sy, start, 1)

  def p1():
    return steps // 2

  def p2():
    pipes = convert_S(sx_sy)
    n = 0
    for x_y, _ in [p for p in pipes.items() if p[0] not in visited]:
      inside = False
      open_ = ''
      for px_py in sorted((p for p in visited if p[1] == x_y[1] and p[0] < x_y[0])):
        if pipes[px_py] == '|':
          inside = not inside
        if pipes[px_py] in {'L', 'F'}:
          open_ = pipes[px_py]
        if pipes[px_py] == '7' and open_ == 'L':
          open_ = ''
          inside = not inside
        if pipes[px_py] == 'J' and open_ == 'F':
          open_ = ''
          inside = not inside
      n += int(inside)
    return n

  return p1(), p2()


def day11():

  def expand_universe(factor, gxs, empty_rows, empty_colums):
    def expand_x(gxs, empty_x):
      return [((x+factor, y) if x > empty_x else (x, y)) for x, y in gxs]
    def expand_y(gxs, empty_y):
      return [((x, y+factor) if y > empty_y else (x, y)) for x, y in gxs]
    return fnt.reduce(expand_y, [y+i*factor for i, y in enumerate(empty_rows)],
                      fnt.reduce(expand_x, [x+i*factor for i, x in enumerate(empty_colums)], gxs))

  def distance(g1, g2):
    return sum((abs(g1[0] - g2[0]), abs(g1[1] - g2[1])))

  rows = [r.strip() for r in get_rows('data/y2023/day11.galaxies.txt')]
  galaxies = [(x, y) for y, row in enumerate(rows) for x, ch in enumerate(row) if ch == '#']
  empty_rows = [y for y, row in enumerate(rows) if set(row) == {'.'}]
  empty_colums = [x for x, col in enumerate(zip(*rows)) if set(col) == {'.'}]

  def p1():
    gxs = expand_universe(1, galaxies, empty_rows, empty_colums)
    return sum(distance(g1, g2) for i, g1 in enumerate(gxs) for g2 in gxs[i+1:])

  def p2():
    gxs = expand_universe(int(1e6) - 1, galaxies, empty_rows, empty_colums)
    return sum(distance(g1, g2) for i, g1 in enumerate(gxs) for g2 in gxs[i+1:])

  return p1(), p2()


def day12():

  rows = [r.strip().split(' ') for r in get_rows('data/y2023/day12.springs.txt')]

  def match(s, g):
    return s[:g].replace('?', '#') == '#'*g and (len(s) == g or s[g] != '#')
  @memoise
  def case_dot(ss, gg, i, j):
    return consume(ss, gg, i, j+1)
  @memoise
  def case_sharp(ss, gg, i, j):
    return consume(ss, gg, i+1, j+gg[i]+1) if match(ss[j:], gg[i]) else 0
  @memoise
  def consume(ss, gg, i, j):
    if i >= len(gg): return int('#' not in ss[j:])
    if j >= len(ss) and i < len(gg): return 0
    return (case_dot(ss, gg, i, j) if ss[j] == '.' else
            case_sharp(ss, gg, i, j) if ss[j] == '#' else
            case_dot(ss, gg, i, j) + case_sharp(ss, gg, i, j))

  def p1():
    return sum(consume(string, tuple(map(int, groups.split(','))), 0, 0) for string, groups in rows)

  def p2():
    return sum(consume('?'.join([string]*5), tuple(map(int, ','.join([groups]*5).split(','))), 0, 0) for string, groups in rows)

  return p1(), p2()


def day13():

  rows = [r.strip() for r in get_rows('data/y2023/day13.mirrors.txt')]

  def split_pattern(ps, row):
    return (ps[:-1] + [ps[-1] + (row,)]) if row else (ps + [()])
  patterns = fnt.reduce(split_pattern, rows[1:], [(rows[0],)])

  @memoise
  def mirrored(pattern, i):
    return all(a == b for a, b in zip(reversed(pattern[:i]), pattern[i:]))

  @memoise
  def mirrors(pattern):
    return ([100*i for i in range(1, len(pattern)) if mirrored(pattern, i)] +
            [i for i in range(1, len(transpose(pattern))) if mirrored(transpose(pattern), i)])

  def p1():
    return sum(mirrors(tuple(pattern))[0] for pattern in patterns)

  def differ_by_1(l1, l2):
    return sum(ch1 == ch2 for ch1, ch2 in zip(l1, l2)) == len(l1)-1

  def smudge(fn, pattern):
    orig = set(mirrors(fn(pattern)))
    for i, p1 in enumerate(pattern):
      for p2 in pattern[i+1:]:
        if differ_by_1(p1, p2):
          ms = set(mirrors(fn(pattern[:i] + (p2,) + pattern[i+1:])))
          if ms - orig:
            return (ms - orig).pop()

  def clean(pattern):
    return smudge(lambda x: x, pattern) or smudge(transpose, transpose(pattern))

  def p2():
    return sum(clean(pattern) for pattern in patterns)

  return p1(), p2()


def day14():

  rows = tuple(r.strip() for r in get_rows('data/y2023/day14.rocks.txt'))

  def move_rock(i, column):
    for j, r in enumerate(column[i+1:]):
      if r == '#':
        return move_rocks(i+j+2, column)
      if r == 'O':
        return move_rocks(i+1, column[:i] + 'O' + column[i+1:j+i+1] + '.' + column[j+i+2:])
    return move_rocks(i+1, column)

  @memoise
  def move_rocks(i, column):
    if i >= len(column):
      return column
    if column[i] != '.':
      return move_rocks(i+1, column)
    for j, r in enumerate(column[i+1:]):
      if r == '#':
        return move_rocks(i+j+2, column)
      if r == 'O':
        return move_rocks(i+1, column[:i] + 'O' + column[i+1:j+i+1] + '.' + column[j+i+2:])
    return column

  def load(pattern):
    return sum((i+1)*row.count('O') for i, row in enumerate(reversed(pattern)))

  def p1():
    moved = transpose(move_rocks(0, column) for column in transpose(rows))
    return load(moved)

  def p2():
    next_d = {'n': 'w', 'w': 's', 's': 'e', 'e': 'n'}
    N = 1000000000 * 4

    def flip(pattern):
      return tuple(map(lambda p: ''.join(reversed(p)), pattern))

    @memoise
    def cycle_rocks(d, pattern):
      def cycle_step(lines):
        return tuple(move_rocks(0, line) for line in lines)
      return (transpose(cycle_step(transpose(pattern))) if d == 'n' else
              cycle_step(pattern) if d == 'w' else
              transpose(flip(cycle_step(flip(transpose(pattern))))) if d == 's' else
              flip(cycle_step(flip(pattern))))

    def cycle(c, d, pattern, seen):
      if c == N:
        return pattern
      pattern = cycle_rocks(d, pattern)
      if (d, pattern) in seen and len(seen[(d, pattern)]) == 2:
        burn, rep = seen[(d, pattern)]
        return cycle(N - (N - burn) % (rep - burn), next_d[d], pattern, {})
      return cycle(c+1, next_d[d], pattern,
                   ((seen | {(d, pattern): [c]}) if (d, pattern) not in seen else
                    (seen | {(d, pattern): seen[(d, pattern)] + [c]})))

    return load(cycle(1, 'n', rows, {}))

  return p1(), p2()


def day15():

  rows = tuple(r.strip() for r in get_rows('data/y2023/day15.hash.txt'))

  def hash(s):
    def hash_step(h, ch):
      return (h + ord(ch)) * 17 % 256
    return fnt.reduce(hash_step, s, 0)

  def p1():
    return sum(hash(seq) for seq in rows[0].split(','))

  def p2():
    def process_lens(boxes, lens):
      if lens.endswith('-'):
        label = lens[:-1]
        box = hash(label)
        return boxes | ({box: tuple((l, f) for l, f in boxes[box] if l != label)} if box in boxes else {})
      label, focus = lens.split('=')
      box = hash(label)
      if box not in boxes:
        return boxes | {box: ((label, int(focus)),)}
      if label in list(map(fst, boxes[box])):
        return boxes | {box: tuple((l, int(focus) if l == label else f) for l, f in boxes[box])}
      return boxes | {box: boxes[box] + ((label, int(focus)),)}
    boxes = fnt.reduce(process_lens, rows[0].split(','), {})
    return sum((box+1) * (i+1) * (f) for box, lenses in boxes.items() for i, (_, f) in enumerate(lenses))

  return p1(), p2()


def day16():

  def p1():
    pass

  def p2():
    pass

  return p1(), p2()


def day17():

  def p1():
    pass

  def p2():
    pass

  return p1(), p2()


def day18():

  def p1():
    pass

  def p2():
    pass

  return p1(), p2()


def day19():

  def p1():
    pass

  def p2():
    pass

  return p1(), p2()


def day20():

  def p1():
    pass

  def p2():
    pass

  return p1(), p2()


def day21():

  def p1():
    pass

  def p2():
    pass

  return p1(), p2()


def day22():

  def p1():
    pass

  def p2():
    pass

  return p1(), p2()


def day23():

  def p1():
    pass

  def p2():
    pass

  return p1(), p2()


def day24():

  def p1():
    pass

  def p2():
    pass

  return p1(), p2()


def day25():

  def p1():
    pass

  def p2():
    pass

  return p1(), p2()


if __name__ == '__main__':
  # print('Day 1 solutions:', day1())
  # print('Day 2 solutions:', day2())
  # print('Day 3 solutions:', day3())
  # print('Day 4 solutions:', day4())
  # print('Day 5 solutions:', day5())
  # print('Day 6 solutions:', day6())
  # print('Day 7 solutions:', day7())
  # print('Day 8 solutions:', day8())
  # print('Day 9 solutions:', day9())
  # print('Day 10 solutions:', day10())
  # print('Day 11 solutions:', day11())
  # print('Day 12 solutions:', day12())
  # print('Day 13 solutions:', day13())
  # print('Day 14 solutions:', day14())
  print('Day 15 solutions:', day15())
  # print('Day 16 solutions:', day16())
  # print('Day 17 solutions:', day17())
  # print('Day 18 solutions:', day18())
  # print('Day 19 solutions:', day19())
  # print('Day 20 solutions:', day20())
  # print('Day 21 solutions:', day21())
  # print('Day 22 solutions:', day22())
  # print('Day 23 solutions:', day23())
  # print('Day 24 solutions:', day24())
  # print('Day 25 solutions:', day25())
