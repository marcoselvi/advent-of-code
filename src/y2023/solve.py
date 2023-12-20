from collections import Counter
import math
import operator as op
import re
import sys

import functools as fnt
import itertools as it

from .. import utils as ut


def day1(lines):
  """https://adventofcode.com/2023/day/1"""

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

  def add_up(digits, lines):
    return sum(digits(line) for line in lines)

  def p1_digits(line):
    numbers = re.findall(r'\d', line)
    return int(numbers[0] + numbers[-1])

  p1 = add_up(p1_digits, lines)

  def p2_digits(line):
    pattern = '|'.join(WORD_TO_DIGIT_MAP.keys())
    numbers = [WORD_TO_DIGIT_MAP.get(maybe_word[0], maybe_word[1])
               for maybe_word in re.findall(rf'(?=({pattern}))|(\d)', line)]
    return int(numbers[0] + numbers[-1])

  p2 = add_up(p2_digits, lines)

  return p1, p2


def day2(lines):
  """https://adventofcode.com/2023/day/2"""

  games = lines

  def game_rounds(game):
    return [[p.strip().split(' ') for p in r.split(',')] for r in game.split(':')[-1].split(';')]

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
  p1 = sum(game_id(game) for game in possible_games(games))

  def power(game):
    rounds = game_rounds(game)
    reds = [int(n) for r in rounds for n, c in r if c == 'red']
    blues = [int(n) for r in rounds for n, c in r if c == 'blue']
    greens = [int(n) for r in rounds for n, c in r if c == 'green']
    return max(reds) * max(blues) * max(greens)
  p2 = sum(power(game) for game in games)

  return p1, p2


def day3(lines):
  """https://adventofcode.com/2023/day/3"""

  def near(xs_y, sx_sy):
    (xs, y), (sx, sy) = xs_y, sx_sy
    return (min(xs)-1 <= sx <= max(xs)+1) and (y-1 <= sy <= y+1)

  def p1_match(numbers, symbols):
    return [n for (n, (xs, y)) in numbers
            if any(near((xs, y), (sx, sy)) for (_, (sx, sy)) in symbols)]
  def p1_line_elements(line):
    def accum_elements(elem_track, i_ch):
      (elements, (digits, xs)), (i, ch) = elem_track, i_ch
      return ((elements, (digits+ch, xs+(i,))) if ch.isdigit() else
              (elements+[(digits, xs), (ch, i)], ('', ())) if ch != '.' and digits else
              (elements+[(ch, i)], ('', ())) if ch != '.' else
              (elements+[(digits, xs)], ('', ())) if digits else
              (elements, ('', ())))
    return fnt.reduce(accum_elements, enumerate(line + '.'), ([], ('', ())))[0]
  def numbers_and_symbols(lines):
    elements = [(elem, (x, j)) for j, line in enumerate(lines) for elem, x in p1_line_elements(line)]
    return ([(int(elem), x_y) for elem, x_y in elements if elem.isdigit()],
            [(elem, x_y) for elem, x_y in elements if not elem.isdigit()])

  p1 = sum(p1_match(*numbers_and_symbols(lines)))

  def p2_match(numbers, maybe_gears):
    gears = []
    for _, (gx, gy) in maybe_gears:
      near_numbers = tuple(filter(lambda n_xs_y: near(n_xs_y[1], (gx, gy)), numbers))
      if len(near_numbers) == 2:
        gears.append((near_numbers[0][0], near_numbers[1][0]))
    return gears
  def p2_line_elements(line):
    def accum_elements(elem_track, i_ch):
      (elements, (digits, xs)), (i, ch) = elem_track, i_ch
      return ((elements, (digits+ch, xs+(i,))) if ch.isdigit() else
              (elements+[(digits, xs), (ch, i)], ('', ())) if ch == '*' and digits else
              (elements+[(ch, i)], ('', ())) if ch == '*' else
              (elements+[(digits, xs)], ('', ())) if digits else
              (elements, ('', ())))
    return fnt.reduce(accum_elements, enumerate(line + '.'), ([], ('', ())))[0]
  def numbers_and_gears(lines):
    elements = [(elem, (x, j)) for j, line in enumerate(lines) for elem, x in p2_line_elements(line)]
    return ([(int(elem), x_y) for elem, x_y in elements if elem.isdigit()],
            [(elem, x_y) for elem, x_y in elements if not elem.isdigit()])
  def gear_ratios(gears):
    return sum(n1*n2 for n1, n2 in gears)

  # schema: (number, ((xs,), y)), (symbol, (x, y))
  p2 = gear_ratios(p2_match(*numbers_and_gears(lines)))

  return p1, p2


def day4(lines):
  """https://adventofcode.com/2023/day/4"""

  def points(line):
    def split(s):
      return filter(None, s.strip().split(' '))
    _, numbers = line.split(':')
    winning, got = numbers.strip().split('|')
    intersection = set(split(winning)) & set(split(got))
    return int(2**(len(intersection) - 1)) if intersection else 0

  p1 = sum(points(line) for line in lines)

  def card_id(line):
    return int(line.split(':')[0].split(' ')[-1].strip())
  def winning_ns(line):
    return set(filter(None, line.split(':')[-1].split('|')[0].strip().split(' ')))
  def got_ns(line):
    return set(filter(None, line.split(':')[-1].split('|')[-1].strip().split(' ')))
  def n_cards(lines):
    cards = [(card_id(line), winning_ns(line), got_ns(line)) for line in lines]
    counter = {cid: 1 for cid, _, _ in cards}
    for cid, win_n, got_n in cards:
      inter = win_n & got_n
      new_cards = cards[cid:(cid+len(inter))]
      counter = counter | {ncid: (counter[ncid] + counter[cid]) for ncid, _, _ in new_cards}
    return sum(counter.values())

  p2 = n_cards(lines)

  return p1, p2


def day5(lines):
  """https://adventofcode.com/2023/day/5"""

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
  def p1_lowest_location(lines):
    seeds = [int(seed.strip()) for seed in lines[0].split(':')[1].strip().split(' ')]
    maps_str = '\n'.join(lines[2:])
    maps = [(map_name(map_str), map_fn(map_str.strip())) for map_str in maps_str.split('\n\n')]
    return min(fnt.reduce(lambda s, m: m[1](s), maps, seed) for seed in seeds)

  p1 = p1_lowest_location(lines)

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
  def p2_lowest_location(lines):
    seed_values = list(map(int, lines[0].split(':')[1].strip().split(' ')))
    seed_ranges = list(zip(seed_values[::2], seed_values[1::2]))
    maps_str = '\n'.join(lines[2:])
    maps = list(map(map_ranges, maps_str.split('\n\n')))
    return min(map(ut.fst, fnt.reduce(ranges_to_ranges, maps, seed_ranges)))

  p2 = p2_lowest_location(lines)

  return p1, p2


def day6(lines):
  """https://adventofcode.com/2023/day/6"""

  def scores(time):
    return [n * (time - n) for n in range(time)]

  def p1_better_times(lines):
    def process_data(line):
      return [int(n.strip()) for n in line.split(':')[1].strip().split(' ') if n]
    races = list(zip(*map(process_data, lines)))
    all_scores = [(best, scores(time)) for time, best in races]
    return fnt.reduce(lambda x, y: x*y, (len([s for s in ss if s > best]) for best, ss in all_scores), 1)

  p1 = p1_better_times(lines)

  def p2_better_times(lines):
    def process_data(line):
      numbers = [n.strip() for n in line.split(':')[1].strip().split(' ')]
      return int(''.join(numbers))
    time, best = process_data(lines[0]), process_data(lines[1])
    return len([s for s in scores(time) if s > best])

  p2 = p2_better_times(lines)

  return p1, p2


def day7(lines):
  """https://adventofcode.com/2023/day/7"""

  def winnings(hand_sorting, lines):
    return sum((i+1)*int(bet) for i, (hand, bet)
               in enumerate(sorted([line.split(' ') for line in lines],
                                   key=hand_sorting)))

  def p1_hand_sorting(hand_bet):
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
    hand, _ = hand_bet
    return (hand_value(hand), card_values(hand))

  p1 = winnings(p1_hand_sorting, lines)

  def p2_hand_sorting(hand_bet):
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
    hand, _ = hand_bet
    return (hand_value(hand), card_values(hand))

  p2 = winnings(p2_hand_sorting, lines)

  return p1, p2


def day8(lines):
  """https://adventofcode.com/2023/day/8"""

  instructions = lines[0]
  graph = {node.strip(): tuple(left_right.strip().strip('(').strip(')').split(', '))
           for node, left_right in [line.split(' = ') for line in lines[2:]]}

  def p1_traverse(lines):
    steps = 0
    node = 'AAA'
    for lr in it.cycle(instructions):
      node = graph[node][0] if lr == 'L' else graph[node][1]
      steps += 1
      if node == 'ZZZ':
        return steps

  p1 = p1_traverse(lines)

  def p2_traverse(lines):
    def unroll(node):
      steps = 0
      for lr in it.cycle(instructions):
        node = graph[node][0] if lr == 'L' else graph[node][1]
        steps += 1
        if node.endswith('Z'):
          return steps
    nodes = [node for node in graph.keys() if node.endswith('A')]
    return math.lcm(*(unroll(node) for node in graph.keys() if node.endswith('A')))

  p2 = p2_traverse(lines)

  return p1, p2


def day9(lines):
  """https://adventofcode.com/2023/day/9"""

  parsed_lines = [list(map(int, line.split(' '))) for line in lines]

  def subseq(seq):
    return [(b - a) for a, b in zip(seq, seq[1:])]
  def next_value(seq):
    return seq[-1] + next_value(subseq(seq)) if not all(e == 0 for e in seq) else 0
  def prev_value(seq):
    return seq[0] - prev_value(subseq(seq)) if not all(e == 0 for e in seq) else 0

  p1 = sum(next_value(line) for line in parsed_lines)

  p2 = sum(prev_value(line) for line in parsed_lines)

  return p1, p2


def day10(lines):
  """https://adventofcode.com/2023/day/10"""

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

  pipes = {(x, y): pipe for y, line in enumerate(lines) for x, pipe in enumerate(line)}

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

  p1 = steps // 2

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

  p2 = n

  return p1, p2


def day11(lines):
  """https://adventofcode.com/2023/day/11"""

  def expand_universe(factor, gxs, empty_lines, empty_colums):
    def expand_x(gxs, empty_x):
      return [((x+factor, y) if x > empty_x else (x, y)) for x, y in gxs]
    def expand_y(gxs, empty_y):
      return [((x, y+factor) if y > empty_y else (x, y)) for x, y in gxs]
    return fnt.reduce(expand_y, [y+i*factor for i, y in enumerate(empty_lines)],
                      fnt.reduce(expand_x, [x+i*factor for i, x in enumerate(empty_colums)], gxs))

  def distance(g1, g2):
    return sum((abs(g1[0] - g2[0]), abs(g1[1] - g2[1])))

  galaxies = [(x, y) for y, line in enumerate(lines) for x, ch in enumerate(line) if ch == '#']
  empty_lines = [y for y, line in enumerate(lines) if set(line) == {'.'}]
  empty_colums = [x for x, col in enumerate(zip(*lines)) if set(col) == {'.'}]

  gxs = expand_universe(1, galaxies, empty_lines, empty_colums)
  p1 = sum(distance(g1, g2) for i, g1 in enumerate(gxs) for g2 in gxs[i+1:])

  gxs = expand_universe(int(1e6) - 1, galaxies, empty_lines, empty_colums)
  p2 = sum(distance(g1, g2) for i, g1 in enumerate(gxs) for g2 in gxs[i+1:])

  return p1, p2


def day12(lines):
  """https://adventofcode.com/2023/day/12"""

  lines = [l.split(' ') for l in lines]

  def match(s, g):
    return s[:g].replace('?', '#') == '#'*g and (len(s) == g or s[g] != '#')
  @ut.memoise
  def case_dot(ss, gg, i, j):
    return consume(ss, gg, i, j+1)
  @ut.memoise
  def case_sharp(ss, gg, i, j):
    return consume(ss, gg, i+1, j+gg[i]+1) if match(ss[j:], gg[i]) else 0
  @ut.memoise
  def consume(ss, gg, i, j):
    if i >= len(gg): return int('#' not in ss[j:])
    if j >= len(ss) and i < len(gg): return 0
    return (case_dot(ss, gg, i, j) if ss[j] == '.' else
            case_sharp(ss, gg, i, j) if ss[j] == '#' else
            case_dot(ss, gg, i, j) + case_sharp(ss, gg, i, j))

  p1 = sum(consume(string, tuple(map(int, groups.split(','))), 0, 0) for string, groups in lines)

  p2 = sum(consume('?'.join([string]*5), tuple(map(int, ','.join([groups]*5).split(','))), 0, 0) for string, groups in lines)

  return p1, p2


def day13(lines):
  """https://adventofcode.com/2023/day/13"""

  transpose = ut.memoise(ut.transpose_strs)

  def split_pattern(ps, line):
    return (ps[:-1] + [ps[-1] + (line,)]) if line else (ps + [()])
  patterns = fnt.reduce(split_pattern, lines[1:], [(lines[0],)])

  @ut.memoise
  def mirrored(pattern, i):
    return all(a == b for a, b in zip(reversed(pattern[:i]), pattern[i:]))

  @ut.memoise
  def mirrors(pattern):
    return ([100*i for i in range(1, len(pattern)) if mirrored(pattern, i)] +
            [i for i in range(1, len(transpose(pattern))) if mirrored(transpose(pattern), i)])

  p1 = sum(mirrors(tuple(pattern))[0] for pattern in patterns)

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

  p2 = sum(clean(pattern) for pattern in patterns)

  return p1, p2


def day14(lines):
  """https://adventofcode.com/2023/day/14"""

  transpose = ut.memoise(ut.transpose_strs)

  lines = tuple(lines)

  def move_rock(i, column):
    for j, r in enumerate(column[i+1:]):
      if r == '#':
        return move_rocks(i+j+2, column)
      if r == 'O':
        return move_rocks(i+1, column[:i] + 'O' + column[i+1:j+i+1] + '.' + column[j+i+2:])
    return move_rocks(i+1, column)

  @ut.memoise
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
    return sum((i+1)*line.count('O') for i, line in enumerate(reversed(pattern)))

  moved = transpose(move_rocks(0, column) for column in transpose(lines))
  p1 = load(moved)

  next_d = {'n': 'w', 'w': 's', 's': 'e', 'e': 'n'}
  N = 1000000000 * 4

  def flip(pattern):
    return tuple(map(lambda p: ''.join(reversed(p)), pattern))

  @ut.memoise
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

  p2 = load(cycle(1, 'n', lines, {}))

  return p1, p2


def day15(lines):
  """https://adventofcode.com/2023/day/15"""

  lines = tuple(lines)

  def hash(s):
    def hash_step(h, ch):
      return (h + ord(ch)) * 17 % 256
    return fnt.reduce(hash_step, s, 0)

  p1 = sum(hash(seq) for seq in lines[0].split(','))

  def process_lens(boxes, lens):
    if lens.endswith('-'):
      label = lens[:-1]
      box = hash(label)
      return boxes | ({box: tuple((l, f) for l, f in boxes[box] if l != label)} if box in boxes else {})
    label, focus = lens.split('=')
    box = hash(label)
    if box not in boxes:
      return boxes | {box: ((label, int(focus)),)}
    if label in list(map(ut.fst, boxes[box])):
      return boxes | {box: tuple((l, int(focus) if l == label else f) for l, f in boxes[box])}
    return boxes | {box: boxes[box] + ((label, int(focus)),)}
  boxes = fnt.reduce(process_lens, lines[0].split(','), {})

  p2 = sum((box+1) * (i+1) * (f) for box, lenses in boxes.items() for i, (_, f) in enumerate(lenses))

  return p1, p2


def day16(lines):
  """https://adventofcode.com/2023/day/16"""

  nodes = {(i, j): ch for j, line in enumerate(lines) for i, ch in enumerate(line)}
  reflections = {('\\', 'd'): 'r',
                 ('\\', 'u'): 'l',
                 ('\\', 'r'): 'd',
                 ('\\', 'l'): 'u',
                 ('/', 'd'): 'l',
                 ('/', 'u'): 'r',
                 ('/', 'r'): 'u',
                 ('/', 'l'): 'd'}

  def move(pos, d):
    x, y = pos
    return (x+1, y) if d == 'r' else (x-1, y) if d == 'l' else (x, y-1) if d == 'u' else (x, y+1)

  def update(path):
    p, d = path
    if nodes[p] == '.' or (nodes[p] == '|' and d in {'u', 'd'}) or (nodes[p] == '-' and d in {'l', 'r'}):
      return [(move(p, d), d)]
    if nodes[p] == '|' and d in {'l', 'r'}:
      return [(move(p, 'u'), 'u'), (move(p, 'd'), 'd')]
    if nodes[p] == '-' and d in {'u', 'd'}:
      return [(move(p, 'l'), 'l'), (move(p, 'r'), 'r')]
    d = reflections[(nodes[p], d)]
    return [(move(p, d), d)]

  def light(start):
    paths, seen = [start], {start}
    while paths:
      paths = [p_d for path in paths for p_d in update(path) if p_d[0] in nodes and p_d not in seen]
      seen |= set(paths)
    return len({p for p, _ in seen})

  p1 = light(((0, 0), 'r'))

  x_max, y_max = max(nodes.keys())
  edges = ([((0, j), 'r') for j in range(y_max+1)] +
           [((x_max, j), 'l') for j in range(y_max+1)] +
           [((i, 0), 'd') for i in range(x_max+1)] +
           [((i, y_max), 'u') for i in range(x_max+1)])

  p2 = max(light(start) for start in edges)

  return p1, p2


def day17(lines):
  """https://adventofcode.com/2023/day/17"""

  blocks = {(i, j): int(n) for j, line in enumerate(lines) for i, n in enumerate(line)}
  max_x, max_y = max(map(ut.fst, blocks)), max(map(ut.snd, blocks))
  fullvisit = max_x * max_y

  directions = {(1, 0), (-1, 0), (0, 1), (0, -1)}

  def move(x_y, dd):
    return tuple(p + d for p, d in zip(x_y, dd))

  def reverse(dd):
    return tuple(-1*p for p in dd)

  def queue_pop(queue):
    return queue[1:], queue[0]

  def queue_push(queue, x):
    for i, (c, y) in enumerate(queue):
      if c > x[0]:
        return queue[:i] + [x] + queue[i:]
    return queue + [x]

  def crucible(min_dist, max_dist):
    queue = [(0, (0, 0, (0, 0)))]
    seen = set()
    costs = {}
    while queue:
      queue, (cost, (x, y, d)) = queue_pop(queue)
      if (x, y) == (max_x, max_y):
        return cost
      if (x, y, d) in seen:
        continue
      seen.add((x, y, d))
      for dd in directions - {d, reverse(d)}:
        cc = 0
        for dist in range(1, max_dist + 1):
          xx, yy = move((x, y), tuple(map(ut.bind(op.mul, dist), dd)))
          if (xx, yy) in blocks:
            cc += blocks[(xx, yy)]
            if dist < min_dist:
              continue
            nc = cost + cc
            if costs.get((xx, yy, dd), 1e100) <= nc:
              continue
            costs[(xx, yy, dd)] = nc
            queue = queue_push(queue, (nc, (xx, yy, dd)))

  p1 = crucible(1, 3)
  p2 = crucible(4, 10)

  return p1, p2


def day18(lines):
  """https://adventofcode.com/2023/day/18"""

  def move(x_y, dd, s):
    return tuple(p + d*s for p, d in zip(x_y, dd))

  dirs = {'R': (1, 0), 'L': (-1, 0), 'D': (0, 1), 'U': (0, -1)}

  def trace_trench(trenches):

    def make_edge(pos_edges, d_s):
      (pos, edges), (dd, step) = pos_edges, d_s
      newpos = move(pos, dirs[dd], step)
      return newpos, edges + [(pos, newpos)]

    _, edges = fnt.reduce(make_edge, trenches, ((0, 0), []))
    xmin, ymin = min(min(x1, x2) for (x1, y1), (x2, y2) in edges), min(min(y1, y2) for (x1, y1), (x2, y2) in edges)
    # recast so xmin, ymin == 0, 0
    return [((x1 - xmin, y1 - ymin), (x2 - xmin, y2 - ymin)) for (x1, y1), (x2, y2) in edges]

  def suttended(edges):
    ymax = max(max(y1, y2) for (x1, y1), (x2, y2) in edges)
    h_edges = [(xy1, xy2) for xy1, xy2 in edges if xy1[1] == xy2[1]]
    return sum((x2 - x1) * (ymax - y1) for (x1, y1), (x2, _) in h_edges)

  def perimeter(edges):
    p = 0
    for (x1, y1), (x2, y2) in edges:
      p += abs(x1 - x2) + abs(y1 - y2)
    return p

  def p1_readline(l):
    d, s, _ = l.split(' ')
    return d, int(s)

  p1_edges = trace_trench(list(map(p1_readline, lines)))

  p1 = suttended(p1_edges) + perimeter(p1_edges) // 2 + 1

  def p2_readline(l):
    rgb = re.match(r'.+\(\#(.+)\)', l).group(1)
    return {'0': 'R', '1': 'D', '2': 'L', '3': 'U'}[rgb[-1]], int(rgb[:-1], 16)

  p2_edges = trace_trench(list(map(p2_readline, lines)))

  p2 = suttended(p2_edges) + perimeter(p2_edges) // 2 + 1

  return p1, p2



def day19(lines):
  """https://adventofcode.com/2023/day/19"""

  def flow_name(f):
    return re.match(r'(.+)\{.+\}', f.strip()).group(1)
  def apply_op(o, v, c):
    return {'>=': op.ge, '<=': op.le, '>': op.gt, '<': op.lt}[o](v, c)
  def apply_rule(rule, p):
    return rule == 'all' or apply_op(rule[1], p[rule[0]], rule[2])
  def flow_rule(r):
    def parse_cond(c):
      m = re.match(r'(\w)([<>])(\d+)', c)
      return m.group(1), m.group(2), int(m.group(3))
    if ':' in r:
      cond, dest = r.split(':')
      return (parse_cond(cond), dest)
    return ('all', r)
  def flow_rules(f):
    return [flow_rule(r) for r in re.match(r'.+{(.+)}', f.strip()).group(1).split(',')]
  def parse_part(p):
    m = re.match(r'\{x=(\d+),m=(\d+),a=(\d+),s=(\d+)\}', p)
    return {'x': int(m.group(1)), 'm': int(m.group(2)), 'a': int(m.group(3)), 's': int(m.group(4))}
  def maybe_accept(flows, part, f):
    if f == 'A':
      return part
    if f == 'R':
      return None
    for rule, dest in flows[f]:
      if apply_rule(rule, part):
        return maybe_accept(flows, part, dest)
  flows_str, parts_str = '\n'.join(lines).split('\n\n')
  flows = {flow_name(f): flow_rules(f) for f in flows_str.split('\n')}
  parts = [parse_part(p) for p in parts_str.split('\n')]
  accepted = list(filter(None, (maybe_accept(flows, part, 'in') for part in parts)))

  p1 = sum(sum(p.values()) for p in accepted)

  def split_gt(k, c, r):
    if apply_op('>', r[k][0], c):
      return r, None
    if apply_op('<=', r[k][1], c):
      return None, r
    return r | {k: (c+1, r[k][1])}, r | {k: (r[k][0], c)}
  def split_lt(k, c, r):
    if apply_op('<', r[k][1], c):
      return r, None
    if apply_op('>=', r[k][0], c):
      return None, r
    return r | {k: (r[k][0], c-1)}, r | {k: (c, r[k][1])}
  def split_range(rule, r):
    if rule == 'all':
      return r, None
    k, o, c = rule
    return split_lt(k, c, r) if o == '<' else split_gt(k, c, r)
  def consume_flow(caught_r, rule_dest):
    (caught, r), (rule, dest) = caught_r, rule_dest
    catch, r = split_range(rule, r)
    return caught + (range_partition(flows, catch, dest) if catch else []), r
  def range_partition(flows, ran, f):
    if f == 'A':
      return [ran]
    if f == 'R':
      return [None]
    out, r = fnt.reduce(consume_flow, flows[f], ([], ran))
    return out

  allowed_ranges = list(filter(
      None, range_partition(
        flows, {'x': (1, 4000), 'm': (1, 4000), 'a': (1, 4000), 's': (1, 4000)}, 'in')))
  p2 = sum(fnt.reduce(op.mul, [(t - b + 1) for b, t in r.values()], 1) for r in allowed_ranges)

  return p1, p2

def day20(lines):
  """https://adventofcode.com/2023/day/20"""

  def build_graph(lines):
    types, outs = {}, {}
    maps = [(tn, dest.split(', ')) for tn, dest in map(lambda l: l.split(' -> '), lines)]
    for tn, to in maps:
      t, n = ('b', tn) if tn == 'broadcaster' else (tn[0], tn[1:])
      types |= {n: t}
      outs |= {n: to}
    ins = {n: tuple(n_i for n_i, dest in outs.items() if n in dest) for n in types}
    return {n: (types[n], ins[n], outs[n]) for n in types} | {'button': ('', (), ('broadcaster',))}

  def initialise_state(graph):
    def init_state(t, ins):
      return 0 if t == '%' else (0,)*len(ins) if t == '&' else None
    return {n: init_state(t, ins) for n, (t, ins, _) in graph.items()}

  def send(graph, n, p):
    return [(n, nn, p, pulse(n, nn, p)) for nn in graph[n][2]]

  @ut.curry(3)
  def pulse(s, n, p, graph, state):
    if n not in graph:
      return 0, state, []
    if graph[n][0] == 'b':
      return p, state, send(graph, n, p)
    if graph[n][0] == '%':
      state = state | {n: int(not state[n]) if not p else state[n]}
      return int(not p and state[n]), state, send(graph, n, int(state[n] and not p)) if not p else []
    if graph[n][0] == '&':
      state = state | {n: tuple((p if inp == s else prev_p) for inp, prev_p in zip(graph[n][1], state[n]))}
      new_p = int(not all(state[n]))
      return new_p, state, send(graph, n, new_p)

  def p1_evolve_graph(graph, h_l_s):
    high, low, state = h_l_s
    low += 1
    ops = send(graph, 'button', 0)
    while ops:
      (_, _, _, o), ops = ops[0], ops[1:]
      h_l, state, nos = o(graph, state)
      high, low, ops = high + h_l*len(nos), low + (not h_l)*len(nos), ops + nos
    return high, low, state

  graph = build_graph(lines)

  high, low, _ = fnt.reduce(lambda h_l_s, _: p1_evolve_graph(graph, h_l_s),
                            range(1000), (0, 0, initialise_state(graph)))

  p1 = high * low

  def p2_evolve_graph(graph, state):
    ops = send(graph, 'button', 0)
    pulses = []
    while ops:
      (s, d, p, o), ops = ops[0], ops[1:]
      _, state, nos = o(graph, state)
      ops = ops + nos
      pulses.append((s, d, p))
    return pulses, state

  state = initialise_state(graph)
  i = 0
  rx_sources = {nn: None for n in graph for nn in graph[n][1] if 'rx' in graph[n][2]}
  while not all(rx_sources.values()):
    i += 1
    pulses, state = p2_evolve_graph(graph, state)
    for (s, d, p) in pulses:
      if p and s in rx_sources and not rx_sources[s]:
        rx_sources[s] = i

  p2 = math.lcm(*rx_sources.values())

  return p1, p2



def day21(lines):
  """https://adventofcode.com/2023/day/21"""
  pass


def day22(lines):
  """https://adventofcode.com/2023/day/22"""
  pass


def day23(lines):
  """https://adventofcode.com/2023/day/23"""
  pass


def day24(lines):
  """https://adventofcode.com/2023/day/24"""
  pass


def day25(lines):
  """https://adventofcode.com/2023/day/25"""
  pass
