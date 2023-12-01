"""Advent of Code 2020 solutions."""

import collections
import functools as fnt
import itertools as itt
import math
import operator as op
import re

import numpy as np

from . import utils


def day1(lines):
  """Solution to https://adventofcode.com/2020/day/1."""
  numbers = [int(line) for line in lines]
  for x, y in itt.combinations(numbers, 2):
    if x+y == 2020:
      part1 = x*y
      break
  for x, y, z in itt.combinations(numbers, 3):
    if x+y+z == 2020:
      part2 = x*y*z
      break
  return part1, part2


def day2(lines):
  """Solution to https://adventofcode.com/2020/day/2."""
  def policy(rule, count, line):
    rl, p = line.split(': ')
    nums, l = rl.strip().split(' ')
    mi, ma = map(int, nums.split('-'))
    return count + (1 if rule(mi, ma, l, p) else 0)
  def rule1(mi, ma, l, p):
    return mi <= p.count(l) <= ma
  def rule2(mi, ma, l, p):
    return (p[mi-1] == l) != (p[ma-1] == l)
  part1 = fnt.reduce(fnt.partial(policy, rule1), lines, 0)
  part2 = fnt.reduce(fnt.partial(policy, rule2), lines, 0)
  return part1, part2


def day3(lines):
  """Solution to https://adventofcode.com/2020/day/3."""
  def trees(x_inc, y_inc, x, y, ls, trs):
    if y >= len(ls):
      return trs
    return trees(x_inc, y_inc, x+x_inc, y+y_inc, ls,
                 trs + (1 if ls[y][x%len(ls[y])] == '#' else 0))
  part1 = trees(3, 1, 0, 0, lines, 0)
  trs = [trees(xi, yi, 0, 0, lines, 0) for xi, yi in
         [(1, 1), (3, 1), (5, 1), (7, 1), (1, 2)]]
  part2 = fnt.reduce(op.mul, trs, 1)
  return part1, part2


def day4(lines):
  """Solution to https://adventofcode.com/2020/day/4."""
  ecl_values = {'amb', 'blu', 'brn', 'gry', 'grn', 'hzl', 'oth'}
  def valid(ps):
    def valid_hgt(hgt):
      val, tag = hgt[:-2], hgt[-2:]
      return (tag in {'cm', 'in'} and
              (tag == 'cm' and 150 <= int(val) <= 193) or
              (tag == 'in' and 59 <= int(val) <= 76))
    def valid_hcl(hcl):
      hsh, val = hcl[:1], hcl[1:]
      return hsh == '#' and len(val) == 6 and re.fullmatch('^[a-f0-9]*$', val)
    def valid_pid(pid):
      return len(pid) == 9 and re.fullmatch('^[0-9]*$', pid)
    return (1920 <= int(ps.get('byr', 0)) <= 2002 and
            2010 <= int(ps.get('iyr', 0)) <= 2020 and
            2020 <= int(ps.get('eyr', 0)) <= 2030 and
            valid_hgt(ps.get('hgt', '')) and
            valid_hcl(ps.get('hcl', '')) and
            ps.get('ecl', '') in ecl_values and
            valid_pid(ps.get('pid', '')))
  def accrue(fields_passports, line):
    fields, passports = fields_passports
    if not line:
      return {}, passports + [fields]
    new_fields = dict([field.split(':') for field in line.split(' ')])
    return {**fields, **new_fields}, passports
  fields, passports = fnt.reduce(accrue, lines, ({}, []))
  if fields:
    passports = passports + [fields]
  keys = {'byr', 'iyr', 'eyr', 'hgt', 'hcl', 'ecl', 'pid'}
  part1 = fnt.reduce(lambda c, p: c + (1 if keys & p.keys() == keys else 0),
                     passports, 0)
  part2 = fnt.reduce(lambda c, p: c + (1 if valid(p) else 0), passports, 0)
  return part1, part2


def day5(lines):
  """Solution to https://adventofcode.com/2020/day/5."""
  def midpoint_plus(ran):
    return math.ceil((ran[0] + ran[1]) / 2)
  def midpoint_minus(ran):
    return math.floor((ran[0] + ran[1]) / 2)
  def rowcol(ran, st):
    if not st and ran[0] == ran[1]:
      return ran[0]
    elif not st:
      raise ValueError(f'{ran}, {st}')
    newran = ((ran[0], midpoint_minus(ran)) if st[0] in ['F', 'L'] else
              (midpoint_plus(ran), ran[1]))
    return rowcol(newran, st[1:])
  def seat(bf, lr):
    return rowcol((0, 127), bf) * 8 + rowcol((0, 7), lr)
  seat_ids = [seat(l[:-3], l[-3:]) for l in lines]
  part1 = max(seat_ids)
  for i, j in utils.conspairs(sorted(seat_ids)):
    if j - i != 1:
      part2 = int((i + j)/2)
      break
  return part1, part2


def day6(lines):
  """Solution to https://adventofcode.com/2020/day/6."""
  def group(groups_g, line):
    groups, g = groups_g
    if not line:
      return groups+[g], []
    return groups, g+[line]
  grouped, g = fnt.reduce(group, lines, ([], []))
  if g:
    grouped += [g]
  part1 = sum([len(set(''.join(group))) for group in grouped])
  part2 = sum([len(set.intersection(*list(map(set, group))))
               for group in grouped])
  return part1, part2


def day7(lines):
  """Solution to https://adventofcode.com/2020/day/7."""
  def clean_bag(string):
    return re.sub(r' ?bags? ?$', '', string.strip()).strip()
  def parse_child(string):
    num, *color = string.split(' ')
    return int(num), ' '.join(color)
  def parse_line(line):
    root, children = line.split(' contain ')
    return (clean_bag(root),
            [] if children == 'no other bags.' else
            [parse_child(clean_bag(ch))
             for ch in children.strip('.').split(', ')])
  def has_child(graph, node, validator):
    def nodes(children):
      return [b for n, b in children]
    children = nodes(graph[node])
    if not children:
      return False
    elif validator in children:
      return True
    return any([has_child(graph, child, validator) for child in children])
  def sum_children(rules, children):
    if not children:
      return 0
    return sum([x + x*sum_children(rules, rules[ch]) for x, ch in children])
  rules = dict([parse_line(line) for line in lines])
  part1 = sum([has_child(rules, node, 'shiny gold') for node in rules])
  part2 = sum_children(rules, rules['shiny gold'])
  return part1, part2


def day8(lines):
  """Solution to https://adventofcode.com/2020/day/8."""
  def start_state():
    return {'accum': 0, 'visit': set(), 'index': 0}
  def update_state(state, inst, n):
    return {'accum': state['accum'] + (n if inst == 'acc' else 0),
            'index': state['index'] + (n if inst == 'jmp' else 1),
            'visit': state['visit'] | {state['index']}}
  def run_program(program, state):
    inst, n = program[state['index']]
    new_state = update_state(state, inst, n)
    if new_state['index'] in new_state['visit']:
      return ('F', state, inst, n)
    if new_state['index'] == len(program):
      return ('S', new_state, inst, n)
    return run_program(program, new_state)
  def all_variations(program):
    def new_program(i, inst):
      newp = list(program)
      newp[i] = (inst, program[i][1])
      return newp
    variations = []
    for i, (inst, _) in enumerate(program):
      if inst == 'jmp':
        variations.append(new_program(i, 'nop'))
      elif inst == 'nop':
        variations.append(new_program(i, 'jmp'))
    return variations
  def parse_inst(line):
    inst, val = line.split(' ')
    return inst, int(val)
  program = [parse_inst(line) for line in lines]
  part1 = run_program(program, start_state())[1]['accum']
  for program in all_variations(program):
    status, state, _, _ = run_program(program, start_state())
    if status == 'S':
      part2 = state['accum']
      break
  return part1, part2


def day9(lines):
  """Solution to https://adventofcode.com/2020/day/9."""
  def all_sums(xs):
    return {x+y for x, y in itt.combinations(xs, 2)}
  def weakness(ns):
    for i in range(25, len(ns)):
      if ns[i] not in all_sums(ns[(i-25):i]):
        return ns[i]
  def breaking(weak, ns):
    for i in range(0, len(ns)):
      for j in range(i+1, len(ns)):
        if sum(ns[i:j]) == weak:
          return min(ns[i:j]), max(ns[i:j])
  ns = [int(x) for x in lines]
  part1 = weakness(ns)
  part2 = sum(breaking(part1, ns))
  return part1, part2


def day10(lines):
  """Solution to https://adventofcode.com/2020/day/10."""
  adapters = sorted([int(x) for x in lines])
  fullchain = [0] + adapters + [max(adapters)+3]
  diffs = [(y-x) for x, y in utils.conspairs(fullchain)]
  part1 = diffs.count(1) * diffs.count(3)
  graph = dict([(x, [c for c in fullchain[i+1:] if c-x <= 3])
                for i, x in enumerate([0] + adapters)])
  computed = {fullchain[-1]: 1}
  for k in reversed(sorted(graph)):
    computed[k] = sum(computed[c] for c in graph[k])
  part2 = computed[0]
  return part1, part2


def day11(lines):
  """Solution to https://adventofcode.com/2020/day/11."""
  maxi, maxj = len(lines[0]), len(lines)
  directions = [(0, 1), (1, 0), (-1, 0), (0, -1),
                (1, -1), (-1, 1), (-1, -1), (1, 1)]
  def count_adj(state, i, j):
    return sum([(i+x >= 0 and i+x < maxi and j+y >= 0 and j+y < maxj
                 and state[j+y][i+x] == '#')
                for x, y in directions])
  def count_inline(state, i, j):
    def occ_dir(dr):
      (dx, dy), (x, y) = dr, dr
      while i+dx < maxi and i+dx >= 0 and j+dy < maxj and j+dy >= 0:
        seat = state[j+dy][i+dx]
        if seat == '#':
          return True
        elif seat == 'L':
          return False
        dx += x
        dy += y
      return False
    return sum([occ_dir(dr) for dr in directions])
  def change(neigh, x, close=4):
    return ('#' if x == 'L' and neigh == 0 else
            'L' if x == '#' and neigh >= close else
            x)
  def update(counter, state, close):
    adj = [[(counter(state, i, j), x) for i, x in enumerate(row)]
           for j, row in enumerate(state)]
    return [''.join([change(a, x, close) for a, x in row]) for row in adj]
  def transform1(state):
    new_state = update(count_adj, state, 4)
    if new_state == state:
      return new_state
    return transform1(new_state)
  def transform2(state):
    new_state = update(count_inline, state, 5)
    if new_state == state:
      return new_state
    return transform2(new_state)
  part1 = sum([row.count('#') for row in transform1(lines)])
  part2 = sum([row.count('#') for row in transform2(lines)])
  return part1, part2


def day12(lines):
  """Solution to https://adventofcode.com/2020/day/12."""
  def go_east(state, val):
    return {**state, **{'x': state['x'] + val}}
  def go_west(state, val):
    return {**state, **{'x': state['x'] - val}}
  def go_north(state, val):
    return {**state, **{'y': state['y'] + val}}
  def go_south(state, val):
    return {**state, **{'y': state['y'] - val}}
  def move(state, instruction):
    def turn(rot, state, val):
      start = rot.index(state['d'])
      return {**state, **{'d': rot[(start + val // 90) % 4]}}
    def turn_left(state, val):
      return turn(['N', 'W', 'S', 'E'], state, val)
    def turn_right(state, val):
      return turn(['N', 'E', 'S', 'W'], state, val)
    def forward(state, val):
      return {'E': go_east, 'W': go_west,
              'N': go_north, 'S': go_south}[state['d']](state, val)
    i, val = instruction[:1], int(instruction[1:])
    return {'F': forward, 'R': turn_right, 'L': turn_left, 'N': go_north,
            'S': go_south, 'E': go_east, 'W': go_west}[i](state, val)
  def move2(state, instruction):
    def py(_, y):
      return y
    def my(_, y):
      return -y
    def px(x, _):
      return x
    def mx(x, _):
      return -x
    def turn(xop, yop):
      def turn_(state):
        return {**state, **{'x': xop(state['x'], state['y']),
                            'y': yop(state['x'], state['y'])}}
      return turn_
    def turn_left(state, val):
      return {90: turn(my, px), 180: turn(mx, my),
              270: turn(py, mx)}[val](state)
    def turn_right(state, val):
      return {90: turn(py, mx), 180: turn(mx, my),
              270: turn(my, px)}[val](state)
    def forward(state, val):
      return {**state, **{'xs': state['xs'] + val*state['x'],
                          'ys': state['ys'] + val*state['y']}}
    i, val = instruction[:1], int(instruction[1:])
    return {'F': forward, 'R': turn_right, 'L': turn_left, 'N': go_north,
            'S': go_south, 'E': go_east, 'W': go_west}[i](state, val)
  position = fnt.reduce(move, lines, {'x': 0, 'y': 0, 'd': 'E'})
  part1 = abs(position['x']) + abs(position['y'])
  position2 = fnt.reduce(move2, lines, {'x': 10, 'y': 1, 'xs': 0, 'ys': 0})
  part2 = abs(position2['xs']) + abs(position2['ys'])
  return part1, part2


def day13(lines):
  """Solution to https://adventofcode.com/2020/day/13."""
  def stop_past(x, b):
    y = 0
    while y < x:
      y += b
    return y
  def advance(c_s, b_off):
    (c, s), (b, off) = c_s, b_off
    return ((c, s*b) if (c+off) % b == 0 else
            advance((c+s, s), (b, off)))
  time = int(lines[0])
  buses = [int(b) for b in lines[1].split(',') if b != 'x']
  times = sorted([(stop_past(time, b), b) for b in buses])
  leave, best = times[0]
  part1 = (leave - time) * best
  gaps = [(int(b), g) for g, b in enumerate(lines[1].split(',')) if b != 'x']
  part2 = fnt.reduce(advance, gaps, (1, 1))[0]
  return part1, part2


def day14(lines):
  """Solution to https://adventofcode.com/2020/day/14."""
  def int2bit(n):
    return '{:036b}'.format(n)
  def update_mem(mask, mem, inst, val):
    def apply_mask(mask, value):
      def apply(m, n):
        return m if m != 'X' else n
      return int(''.join(map(apply, mask, int2bit(value))), 2)
    addr = int(inst.strip('mem[').strip(']'))
    return {**mem, **{addr: apply_mask(mask, int(val))}}
  def update_mem2(mask, mem, inst, val):
    def addbit(addrs, b):
      return [add + b for add in addrs]
    def branching(addrs, m_a):
      m, a = m_a
      return (addbit(addrs, a) if m == '0' else
              addbit(addrs, '1') if m == '1' else
              addbit(addrs, '0') + addbit(addrs, '1'))
    def addresses(mask, addr):
      return fnt.reduce(branching, zip(mask, int2bit(addr)), [''])
    addr = int(inst.strip('mem[').strip(']'))
    return {**mem, **{int(add, 2): int(val) for add in addresses(mask, addr)}}
  def parse(updater, mem_mask, line):
    mem, mask = mem_mask
    inst, val = line.split(' = ')
    return ((mem, val) if inst == 'mask' else
            (updater(mask, mem, inst, val), mask))
  mem = fnt.reduce(fnt.partial(parse, update_mem), lines, ({}, 'X'*36))[0]
  part1 = sum(mem.values())
  mem2 = fnt.reduce(fnt.partial(parse, update_mem2), lines, ({}, 'X'*36))[0]
  part2 = sum(mem2.values())
  return part1, part2


def day15(lines):
  """Solution to https://adventofcode.com/2020/day/15."""
  ns = [int(n) for n in lines[0].split(',')]
  seq = ns[:-1]
  log = {s: i for i, s in enumerate(seq)}
  new = ns[-1]
  for i in range(len(seq), 2020):
    latest = new
    new = (i - log[latest]) if latest in log else 0
    log[latest] = i
  part1 = latest
  for i in range(log[latest]+1, 30000000):
    latest = new
    new = (i - log[latest]) if latest in log else 0
    log[latest] = i
  part2 = latest
  return part1, part2


def day16(lines):
  """Solution to https://adventofcode.com/2020/day/16."""
  def new_rule(line):
    def parse_ranges(rs):
      vals = set()
      for r in rs:
        mi, ma = r.split('-')
        vals = vals | set(range(int(mi), int(ma)+1))
      return vals
    name, ranges = line.split(': ')
    return name, parse_ranges(ranges.split(' or '))
  def ticket(line):
    return [int(n) for n in line.split(',')]
  def parse_line(state, line):
    if line == 'your ticket:' or line == 'nearby tickets:':
      return state
    st, rules, myt, othts = state
    return (('mt', rules, myt, othts) if not line and st == 'rules' else  # pylint: disable=g-long-ternary
            ('ots', rules, myt, othts) if not line and st == 'mt' else  # pylint: disable=g-long-ternary
            (st, rules + [new_rule(line)], myt, othts) if st == 'rules' else  # pylint: disable=g-long-ternary
            (st, rules, ticket(line), othts) if st == 'mt' else
            (st, rules, myt, othts + [ticket(line)]) if st == 'ots' else
            utils.raise_(ValueError()))
  def valid_fields(rules, x):
    return {r for r, ran in rules if x in ran}
  def find_unique(fields):
    for i, f in enumerate(fields):
      uniqueness = f - set.union(*fields[:i], *fields[i+1:])
      if len(uniqueness) == 1:
        return i, uniqueness.pop()
  def reduce_fields(ordered_fields, idx_field):
    if len(idx_field) == len(ordered_fields):
      return idx_field
    i, f = find_unique(ordered_fields)
    ordered_fields = [of - {f} for of in ordered_fields]
    ordered_fields[i] = set()
    idx_field[i] = f
    return reduce_fields(ordered_fields, idx_field)
  _, rules, mt, ots = fnt.reduce(parse_line, lines, ('rules', [], None, []))
  all_ranges = fnt.reduce(set.union, map(utils.snd, rules), set())
  part1 = sum([x for t in ots for x in t if x not in all_ranges])  # pylint: disable=g-complex-comprehension
  filtered_ots = [ot for ot in ots if all([x in all_ranges for x in ot])]
  ots_fields = [[valid_fields(rules, x) for x in ot] for ot in filtered_ots]
  ordered_fields = [set.intersection(*fields) for fields in zip(*ots_fields)]
  lookup = reduce_fields(ordered_fields, {})
  departures = [mt[i] for i, f in lookup.items() if f.startswith('departure')]
  part2 = fnt.reduce(op.mul, departures, 1)
  return part1, part2


def day17(lines):
  """Solution to https://adventofcode.com/2020/day/17."""
  @utils.memoise
  def neigh3(p):
    x, y, z = p
    return {(x+dx, y+dy, z+dz)
            for dx, dy, dz in itt.product(*[[-1, 0, 1]]*3)} - {p}
  @utils.memoise
  def neigh4(p):
    x, y, z, q = p
    return {(x+dx, y+dy, z+dz, q+dq)
            for dx, dy, dz, dq in itt.product(*[[-1, 0, 1]]*4)} - {p}
  def ndim_conway(all_neighbours, init_state):
    def active_neighbours(state, p):
      return all_neighbours(p) & state
    def update_inactive(state):
      useful = fnt.reduce(set.union, map(all_neighbours, state), set()) - state
      return {p for p in useful if len(active_neighbours(state, p)) == 3}
    def update_universe(state, _):
      new_state = {p for p in state
                   if len(active_neighbours(state, p)) in {2, 3}}
      return new_state | update_inactive(state)
    return fnt.reduce(update_universe, range(6), init_state)
  init3 = {(i, j, 0) for j, line in enumerate(reversed(lines))  # pylint: disable=g-complex-comprehension
           for i, x in enumerate(line) if x == '#'}
  init4 = {(i, j, 0, 0) for j, line in enumerate(reversed(lines))  # pylint: disable=g-complex-comprehension
           for i, x in enumerate(line) if x == '#'}
  part1 = len(ndim_conway(neigh3, init3))
  part2 = len(ndim_conway(neigh4, init4))
  return part1, part2


def day18(lines):
  """Solution to https://adventofcode.com/2020/day/18."""
  def parse_line(line):
    def maybe_separate(ch):
      if ch in {'+', '*'}:
        return [ch]
      elif ch.endswith(')'):
        n, pars = re.match(r'([0-9]+)([\)]+)', ch, re.I).groups()
        return [int(n)] + list(pars)
      elif ch.startswith('('):
        pars, n = re.match(r'([\(]+)([0-9]+)', ch, re.I).groups()
        return list(pars) + [int(n)]
      else:
        return [int(ch)]
    return [x for y in line.split(' ') for x in maybe_separate(y)]  # pylint: disable=g-complex-comprehension
  def split_on_pars(eq, seq):
    if not eq:
      return seq
    if eq[0] == ')':
      return eq[1:], seq
    if eq[0] == '(':
      eq, subs = split_on_pars(eq[1:], [])
      return split_on_pars(eq, seq + [subs])
    return split_on_pars(eq[1:], seq + [eq[0]])
  dos = {'*': op.mul, '+': op.add}
  def consume3(eq):
    n1, do, n2 = eq[:3]
    return [dos[do](n1, n2)] + eq[3:]
  def collapse1(eq):
    if len(eq) == 1:
      return eq[0]
    return collapse1(consume3(eq))
  def collapse2(eq):
    if len(eq) == 1:
      return eq[0]
    if '+' in eq:
      i = eq.index('+')
      return collapse2(eq[:i-1] + [dos['+'](eq[i-1], eq[i+1])] + eq[i+2:])
    return collapse2(consume3(eq))
  def evaluate(collapse, eq):
    return collapse([(evaluate(collapse, x) if isinstance(x, list) else x)
                     for x in eq])
  eqs = [split_on_pars(parse_line(line), []) for line in lines]
  part1 = sum([evaluate(collapse1, eq) for eq in eqs])
  part2 = sum([evaluate(collapse2, eq) for eq in eqs])
  return part1, part2


def day19(lines):
  """Solution to https://adventofcode.com/2020/day/19."""
  rs, ms = ','.join(lines).split(',,')
  rules = {k: rule for k, rule in
           [r.split(': ') for r in rs.split(',')]}
  messages = ms.split(',')
  @utils.setmemoise
  def rulejoin(*resolved_rules):
    return {''.join(rlprod) for rlprod in itt.product(*resolved_rules)}
  @utils.memoise
  def resolve(rule):
    if '"' in rule:
      return {rule.strip('"')}
    resrules = [[resolve(rules[y]) for y in x.split(' ')]
                for x in rule.split(' | ')]
    return fnt.reduce(set.union, [rulejoin(*ress) for ress in resrules], set())
  def match_part2(m):
    rule42 = resolve(rules['42'])
    rule31 = resolve(rules['31'])
    if len(m) % 8 != 0 or len(m) < 24:
      return False
    def match11(r):
      return len(r) >= 16 and r[:8] in rule42 and r[-8:] in rule31
    def match8(r):
      return r[:8] in rule42
    def consume_chunk(r):
      return r[8:-8] if match11(r) else r[8:] if match8(r) else None
    rest = m[8:] if match8(m) else None
    if rest is None:
      return False
    rest = rest[8:-8] if match11(rest) else None
    if rest is None:
      return False
    while rest:
      rest = consume_chunk(rest)
      if rest is None:
        return False
    return True
  rule0 = resolve(rules['0'])
  part1 = len(list(filter(lambda m: m in rule0, messages)))
  part2 = len(list(filter(match_part2, messages)))
  return part1, part2


def day20(lines):
  """Solution to https://adventofcode.com/2020/day/20."""
  flip_x = fnt.partial(np.flip, axis=1)
  flip_y = fnt.partial(np.flip, axis=0)
  rot180 = utils.compose(np.rot90, np.rot90)
  rot270 = utils.compose(rot180, np.rot90)
  transformations = [utils.identity, flip_x, flip_y, np.rot90, rot180, rot270,
                     utils.compose(flip_x, np.rot90),
                     utils.compose(flip_y, np.rot90)]
  def edg(*eds):
    return {'t': eds[0], 'r': eds[1], 'b': eds[2], 'l': eds[3]}
  def make_edges(im):
    return edg(*list(map(''.join, [im[0, :], im[:, -1], im[-1, :], im[:, 0]])))
  def parse_tile(tile):
    title, *rows = tile.split(',')
    tile = np.array([list(row) for row in rows])
    variations = [trans(tile) for trans in transformations]
    edges = [make_edges(var) for var in variations]
    return (int(title.strip('Tile ').strip(':')), variations, edges)
  def rm_tile(tiles, rem):
    return [(t, variations, edges) for t, variations, edges in tiles
            if t != rem]
  def inc_pos(pos, maxlen):
    x, y = pos
    return (x+1, y) if x+1 < maxlen else (0, y+1)
  def match(assigned, pos, o):
    x, y = pos
    m1 = assigned[(x-1, y)][2]['r'] == o['l'] if x > 0 else True
    m2 = assigned[(x, y-1)][2]['b'] == o['t'] if y > 0 else True
    return m1 and m2
  def solvejigsaw(ass, pos, tiles, edgelen):
    if not tiles:
      return ass
    for t, variations, edges in tiles:
      for var, edg in zip(variations, edges):
        if match(ass, pos, edg):
          puzzle = solvejigsaw({**ass, **{pos: (t, var, edg)}},
                               inc_pos(pos, edgelen), rm_tile(tiles, t),
                               edgelen)
          if puzzle is not None:
            return puzzle
  tiles = [parse_tile(tile) for tile in ','.join(lines).split(',,')]
  maxlen = int(math.sqrt(len(tiles)))
  solved = solvejigsaw({}, (0, 0), tiles, edgelen=maxlen)
  corners = (solved[(0, 0)][0], solved[(0, maxlen-1)][0],
             solved[(maxlen-1, 0)][0], solved[(maxlen-1, maxlen-1)][0])
  part1 = fnt.reduce(op.mul, corners, 1)
  monsterinp = ['                  # ',
                '#    ##    ##    ###',
                ' #  #  #  #  #  #   ']
  monster = np.array([list(row) for row in monsterinp])
  monsteridxs = np.where(monster == '#')
  def prune_edges(im):
    return im[1:-1, 1:-1]
  def assemble_image(pr_tls):
    lookup = dict(pr_tls)
    ordered = [[lookup[(i, j)] for i in range(maxlen)]
               for j in range(maxlen)]
    return np.concatenate([np.concatenate(row, axis=1) for row in ordered],
                          axis=0)
  full_image = assemble_image([(pos, prune_edges(im)) for pos, (t, im, edg)
                               in solved.items()])
  def count_monsters(im):
    monsters = []
    for i in range(len(im) - 19):
      for j in range(len(im) - 2):
        if np.all(im[monsteridxs[0]+j, monsteridxs[1]+i] == '#'):
          monsters += [(i, j)]
    return monsters
  def remove_monsters(indices, im):
    for i, j in indices:
      im[monsteridxs[0]+j, monsteridxs[1]+i] = '.'
    return im
  how_many = [(count_monsters(trans(full_image)), trans(full_image))
              for trans in transformations]
  indices, im = list(sorted(how_many, reverse=True, key=lambda x: len(x[0])))[0]
  demonstered = remove_monsters(indices, im)
  part2 = dict(zip(*np.unique(demonstered, return_counts=True)))['#']
  return part1, part2


def day21(lines):
  """Solution to https://adventofcode.com/2020/day/21."""
  def parse_line(line):
    ingreds, allergs = line.split(' (')
    allergs = allergs.strip(')').replace('contains ', '').split(', ')
    return ingreds.split(' '), allergs
  foods = [parse_line(line) for line in lines]
  allingreds = [ingred for ingreds, _ in foods for ingred in ingreds]  # pylint: disable=g-complex-comprehension
  def allergen_inventory(inventory, allergen):
    could_contain = fnt.reduce(
        set.intersection,
        [set(ingreds) for ingreds, allergs in foods if allergen in allergs],
        set(allingreds))
    return {**inventory,
            **{cc: inventory.get(cc, set()) | {allergen}
               for cc in could_contain}}
  def allergic(inventory, l):
    def remove_al(inv, al):
      return {k: (v - {al}) for k, v in inv.items() if v - {al} != set()}
    if not inventory:
      return l
    unique, al = [(k, v) for k, v in inventory.items() if len(v) == 1][0]
    al = al.pop()
    return allergic(remove_al(inventory, al), l + [(al, unique)])
  allallergs = {allerg for _, allergs in foods for allerg in allergs}  # pylint: disable=g-complex-comprehension
  allerginvent = fnt.reduce(allergen_inventory, allallergs, {})
  okingreds = {ingred for ingred in allingreds if ingred not in allerginvent}
  part1 = sum([allingreds.count(okingred) for okingred in okingreds])
  part2 = ','.join(map(utils.snd, sorted(allergic(allerginvent, []))))
  return part1, part2


def day22(lines):
  """Solution to https://adventofcode.com/2020/day/22."""
  def parseplayer(vals):
    return list(map(int, vals.split(',')[1:]))
  def play(p1, p2):
    if not p1:
      return p2
    if not p2:
      return p1
    top1, top2 = p1[0], p2[0]
    return play(p1[1:] + ([top1, top2] if top1 > top2 else []),
                p2[1:] + ([top2, top1] if top2 > top1 else []))
  def playrec(p1, p2, seen):
    if not p1:
      return 'p2', p2
    if not p2:
      return 'p1', p1
    rhash = (tuple(p1), tuple(p2))
    if rhash in seen:
      return 'p1', p1
    top1, top2 = p1[0], p2[0]
    if len(p1[1:]) >= top1 and len(p2[1:]) >= top2:
      win, _ = playrec(p1[1:top1+1], p2[1:top2+1], set())
      return playrec(p1[1:] + ([top1, top2] if win == 'p1' else []),
                     p2[1:] + ([top2, top1] if win == 'p2' else []),
                     seen | {rhash})
    else:
      return playrec(p1[1:] + ([top1, top2] if top1 > top2 else []),
                     p2[1:] + ([top2, top1] if top2 > top1 else []),
                     seen | {rhash})
  def score(cards):
    return sum([(i+1)*x for i, x in enumerate(reversed(cards))])
  player1, player2 = map(parseplayer, ','.join(lines).split(',,'))
  winner1 = play(player1, player2)
  part1 = score(winner1)
  _, winner2 = playrec(player1, player2, set())
  part2 = score(winner2)
  return part1, part2


def day23(lines):
  """Solution to https://adventofcode.com/2020/day/23."""
  def makecups(l):
    return {**{a: b for a, b in utils.conspairs(l)}, **{l[-1]: l[0]}}
  def printer(cups, s_i, _):
    s, i = s_i
    s += str(cups[i])
    return s, cups[i]
  def playround(maxval, state, _):
    cups, curr = state
    p1 = cups[curr]
    p2 = cups[p1]
    p3 = cups[p2]
    cups[curr] = cups[p3]
    dest = curr - 1 or maxval
    while dest in [p1, p2, p3]:
      dest = dest - 1 or maxval
    cups[p3] = cups[dest]
    cups[p2] = p3
    cups[p1] = p2
    cups[dest] = p1
    return cups, cups[curr]
  initcups = list(map(int, lines[0]))
  newcups, _ = fnt.reduce(fnt.partial(playround, 9),
                          range(100), (makecups(initcups), initcups[0]))
  part1 = int(fnt.reduce(fnt.partial(printer, newcups),
                         range(1, 9), ('', 1))[0])
  longcups = initcups + list(range(10, 1000001))
  newcups2, _ = fnt.reduce(fnt.partial(playround, int(1e6)),
                           range(10000000), (makecups(longcups), longcups[0]))
  part2 = newcups2[1] * newcups2[newcups2[1]]
  return part1, part2


def day24(lines):
  """Solution to https://adventofcode.com/2020/day/24."""
  def parse_moves(s, ms):
    if not s:
      return ms
    return (parse_moves(s[1:], ms + [s[0]]) if s[0] in {'e', 'w'} else
            parse_moves(s[2:], ms + [s[:2]]))
  ops = {'e': lambda tile: (tile[0]+1, tile[1]-1, tile[2]),
         'se': lambda tile: (tile[0], tile[1]-1, tile[2]+1),
         'sw': lambda tile: (tile[0]-1, tile[1], tile[2]+1),
         'w': lambda tile: (tile[0]-1, tile[1]+1, tile[2]),
         'nw': lambda tile: (tile[0], tile[1]+1, tile[2]-1),
         'ne': lambda tile: (tile[0]+1, tile[1], tile[2]-1)}
  def id_tile(tile, m):
    return ops[m](tile)
  @utils.memoise
  def allneighs(tile):
    return {op(tile) for op in ops.values()}
  def update_black(bts):
    return {b for b in bts if len(allneighs(b) & bts) in {1, 2}}
  def update_white(bts):
    wts = {t for b in bts for t in allneighs(b)} - bts  # pylint: disable=g-complex-comprehension
    return {w for w in wts if len(allneighs(w) & bts) == 2}
  def evolve(black_tiles, _):
    return update_black(black_tiles) | update_white(black_tiles)
  moves = [parse_moves(line, []) for line in lines]
  flips = [fnt.reduce(id_tile, ms, (0, 0, 0)) for ms in moves]
  black_tiles = {tile for tile, c in collections.Counter(flips).items()
                 if c % 2}
  part1 = len(black_tiles)
  part2 = len(fnt.reduce(evolve, range(100), black_tiles))
  return part1, part2


def day25(lines):
  """Solution to https://adventofcode.com/2020/day/25."""
  def trans(subjn, n):
    return (n*subjn) % 20201227
  def loopsize(subjn, key):
    i, j = 1, 0
    while i != key:
      i = trans(subjn, i)
      j += 1
    return j
  cardkey, doorkey = int(lines[0]), int(lines[1])
  cardloop = loopsize(7, cardkey)
  part1 = fnt.reduce(lambda x, _: trans(doorkey, x), range(cardloop), 1)
  return part1
