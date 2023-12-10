"""Solutions to Advent of Code 2019."""

import fractions
import functools as fnt
import itertools as itt
import math

from .. import utils


# --- INTCODE COMPUTER ---
def parse_intcode_mem(text):
  return [int(x) for x in text.strip().split(',')]


def start_state(text, inp):
  return {'mem': parse_intcode_mem(text), 'i': 0, 'out': 0, 'inp': inp,
          'relbase': 0, 'status': 'init'}


def elastic_mem(state, i):
  if i >= len(state['mem']):
    state['mem'] = state['mem'] + ([0] * (i+1-len(state['mem'])))
  return state


def update_mem(state, i, v):
  state = elastic_mem(state, i)
  state['mem'][i] = v
  return state


def read_mem(state, i):
  state = elastic_mem(state, i)
  return state['mem'][i]


def cons_mem(f, state, off):
  return read_mem(state, f(state, state['i']+off))


def update_i(state, i):
  return {**state, **{'i': i}}


def update_out(state, out):
  return {**state, **{'out': out}}


def append_inp(state, inp):
  return {**state, **{'inp': state['inp'] + [inp]}}


def consume_inp(state):
  x = state['inp'][0]
  return x, {**state, **{'inp': state['inp'][1:]}}


def update_relbase(state, x):
  return {**state, **{'relbase': state['relbase']+x}}


def update_status(state, status):
  state['status'] = status
  return state


def get_op(m):
  def pos(state, j):
    return read_mem(state, j)
  def imm(_, j):
    return j
  def rel(state, j):
    return read_mem(state, j) + state['relbase']
  return {'2': rel, '1': imm, '0': pos}[m]


def get_mode_op(mode_op):
  mode_op = '0'*(5 - len(mode_op)) + mode_op
  mode, op = mode_op[:-2], mode_op[-2:]
  return [get_op(m) for m in reversed(mode)], op


def intcode(state):
  """Intcode Computer."""
  def add(state, a1, a2, o):
    new_state = update_mem(state, o(state, state['i']+3),
                           cons_mem(a1, state, 1) + cons_mem(a2, state, 2))
    return update_status(update_i(new_state, state['i']+4), 'cont')
  def mul(state, m1, m2, o):
    new_state = update_mem(state, o(state, state['i']+3),
                           cons_mem(m1, state, 1) * cons_mem(m2, state, 2))
    return update_status(update_i(new_state, state['i']+4), 'cont')
  def ass(state, x, unused_y, unused_z):
    i = state['i']
    inp, new_state = consume_inp(state)
    return update_status(update_i(update_mem(new_state, x(state, i+1), inp),
                                  i+2),
                         'cont')
  def jit(state, c, v, _):
    if cons_mem(c, state, 1):
      return update_status(update_i(state, cons_mem(v, state, 2)), 'cont')
    return update_status(update_i(state, state['i']+3), 'cont')
  def jif(state, c, v, _):
    if not cons_mem(c, state, 1):
      return update_status(update_i(state, cons_mem(v, state, 2)), 'cont')
    return update_status(update_i(state, state['i']+3), 'cont')
  def lth(state, l1, l2, p):
    mem_update = int(cons_mem(l1, state, 1) < cons_mem(l2, state, 2))
    new_state = update_mem(state, p(state, state['i']+3), mem_update)
    return update_status(update_i(new_state, state['i']+4), 'cont')
  def equ(state, e1, e2, p):
    mem_update = int(cons_mem(e1, state, 1) == cons_mem(e2, state, 2))
    new_state = update_mem(state, p(state, state['i']+3), mem_update)
    return update_status(update_i(new_state, state['i']+4), 'cont')
  def rlb(state, x, unused_y, unused_z):
    return update_status(update_i(update_relbase(state, cons_mem(x, state, 1)),
                                  state['i']+2),
                         'cont')
  def out(state, x, unused_y, unused_z):
    return update_status(update_i(update_out(state, cons_mem(x, state, 1)),
                                  state['i']+2),
                         'break')
  def exi(state, *_):
    return update_status(state, 'exit')
  mode, op = get_mode_op(str(read_mem(state, state['i'])))
  return {'01': add, '02': mul, '03': ass, '04': out,
          '05': jit, '06': jif, '07': lth, '08': equ,
          '09': rlb,
          '99': exi}[op](state, *mode)


def run_intcode(state, term=['break'], verbose=False):
  while state['status'] not in term:
    state = intcode(state)
    if verbose and state['status'] == 'break':
      print(state['out'])
  return state


# --- TASKS ---
def day1(lines):
  """Solution to https://adventofcode.com/2019/day/1."""
  def fuel(x):
    return math.floor(int(x) / 3) - 2
  def add_fuel(f):
    req = fuel(f)
    while req > 0:
      f += req
      req = fuel(req)
    return f
  fuels = [fuel(int(x)) for x in lines]
  part1 = sum(fuels)
  part2 = sum([add_fuel(f) for f in fuels])
  return part1, part2


def day2(lines):
  """Solution to https://adventofcode.com/2019/day/2."""
  def start():
    return [int(x) for x in lines[0].strip().split(',')]
  def run(comp, i):
    if comp[i] == 1:
      comp[comp[i+3]] = comp[comp[i+1]] + comp[comp[i+2]]
      return run(comp, i+4)
    elif comp[i] == 2:
      comp[comp[i+3]] = comp[comp[i+1]] * comp[comp[i+2]]
      return run(comp, i+4)
    elif comp[i] == 99:
      return comp
    else:
      raise ValueError()
  comp = start()
  comp[1] = 12
  comp[2] = 2
  part1 = run(comp, 0)[0]
  for x, y in itt.product(range(0, 100), range(0, 100)):
    comp = start()
    comp[1] = x
    comp[2] = y
    if run(comp, 0)[0] == 19690720:
      part2 = 100*x + y
      break
  return part1, part2


def day3(lines):
  """Solution to https://adventofcode.com/2019/day/3."""
  def move(sx, sy, i, d):
    return ((sx + i + 1, sy) if d == 'R' else (sx - i - 1, sy) if d == 'L' else
            (sx, sy + i + 1) if d == 'U' else (sx, sy - i - 1))
  def trace(start, d, n):
    sx, sy = start
    return [move(sx, sy, i, d) for i in range(n)]
  def all_locs(locs, step):
    d, n = step[:1], step[1:]
    if not locs:
      return trace((0, 0), d, int(n))
    return locs + trace(locs[-1], d, int(n))
  path1 = fnt.reduce(all_locs, lines[0].strip().split(','), [])
  path2 = fnt.reduce(all_locs, lines[1].strip().split(','), [])
  inters = set(path1) & set(path2)
  part1 = min([abs(ix) + abs(iy) for ix, iy in inters])
  part2 = min([path1.index(i) + path2.index(i) + 2 for i in inters])
  return part1, part2


def day4(lines):
  """Solution to https://adventofcode.com/2019/day/4."""
  def valid(x):
    return len(set(x)) < len(x) and ''.join(sorted(x)) == x
  def single_pair(pairs):
    eqp = [x == y for x, y in pairs]
    len_1 = len(eqp) - 1
    return any([eqp[i] and (i == 0 or not eqp[i-1])
                and (i == len_1 or not eqp[i+1])
                for i in range(len(eqp))])
  def valid2(x):
    return valid(x) and single_pair(utils.conspairs(x))
  mi, ma = lines[0].strip().split('-')
  allps = [str(x) for x in range(int(mi), int(ma))]
  part1 = len(list(filter(valid, allps)))
  part2 = len(list(filter(valid2, allps)))
  return part1, part2


def day5(lines):
  """Solution to https://adventofcode.com/2019/day/5."""
  part1 = run_intcode(start_state(lines[0], [1]), term=['exit'])['out']
  part2 = run_intcode(start_state(lines[0], [5]), term=['exit'])['out']
  return part1, part2


def day6(lines):
  """Solution to https://adventofcode.com/2019/day/6."""
  def parents(searcher, k, ps):
    if k == 'COM':
      return ps
    return parents(searcher, searcher[k], ps+1)
  def visits(searcher, k, st, vs):
    if k == 'COM':
      return {**vs, **{'COM': st}}
    return visits(searcher, searcher[k], st+1, {**vs, **{k: st}})
  orbits = [line.split(')') for line in lines]
  searcher = {y: x for x, y in orbits}
  part1 = sum([parents(searcher, k, 0) for k in searcher])
  you_visits = visits(searcher, 'YOU', 0, {})
  san_visits = visits(searcher, 'SAN', 0, {})
  part2 = min([you_visits[k] + san_visits[k] - 2 for k in
               you_visits.keys() & san_visits.keys()])
  return part1, part2


def day7(lines):
  """Solution to https://adventofcode.com/2019/day/7."""
  def amplifier_chain(phases):
    def amplifier(inp, phase):
      return run_intcode(start_state(lines[0], [phase, inp]),
                         term=['exit'])['out']
    return fnt.reduce(amplifier, phases, 0)
  def amplifier_loop(phases):
    def amplifier(state, inp):
      state = run_intcode(append_inp(update_status(state, 'cont'), inp),
                          term=['break', 'exit'])
      return state, state['out']
    def one_loop(states, inp):
      return utils.map_accum(amplifier, states, inp)
    states = [start_state(lines[0], [phase]) for phase in phases]
    inp = 0
    while states[-1]['status'] != 'exit':
      states, inp = one_loop(states, inp)
    return states[-1]['out']
  part1 = max([amplifier_chain(phases) for phases in
               itt.permutations(list(range(5)))])
  part2 = max([amplifier_loop(phases) for phases in
               itt.permutations(list(range(5, 10)))])
  return part1, part2


def day8(lines):
  """Solution to https://adventofcode.com/2019/day/8."""
  def select_color(ns):
    def only_color(prev, n):
      return n if prev == '2' else prev
    return fnt.reduce(only_color, ns, '2')
  digits = lines[0]
  layers = [digits[i:(i+25*6)] for i in range(0, len(digits), 25*6)]
  fewest_0s = sorted([(l.count('0'), l) for l in layers])[0][1]
  part1 = fewest_0s.count('1') * fewest_0s.count('2')
  combined = ''.join(list(map(select_color, zip(*layers)))).replace('0', ' ')
  part2 = '\n'.join([combined[i:i+25] for i in range(0, len(combined), 25)])
  return part1, part2


def day9(lines):
  """Solution to https://adventofcode.com/2019/day/9."""
  part1 = run_intcode(start_state(lines[0], [1]),
                      term=['exit'], verbose=False)['out']
  part2 = run_intcode(start_state(lines[0], [2]),
                      term=['exit'], verbose=False)['out']
  return part1, part2


def day10(lines):
  """Solution to https://adventofcode.com/2019/day/10."""
  sign = fnt.partial(math.copysign, 1)
  def dist(b, a):
    (xb, yb), (xa, ya) = b, a
    return (xb-xa), (yb-ya)
  def drctn(b, a):
    dx, dy = dist(b, a)
    if dy == 0:
      return (sign(dx), 0)
    elif dx == 0:
      return (0, sign(dy))
    else:
      f = fractions.Fraction(dy, dx)
      return (sign(dx)*abs(int(f.denominator)), sign(dy)*abs(int(f.numerator)))
  def angle(d):
    x, y = d
    angle = math.atan2(x, -y)
    return angle if angle >= 0 else 2*math.pi + angle
  asteroids = [(i, j) for i in range(len(lines[0])) for j in range(len(lines))
               if lines[j][i] == '#']
  distances = [(a, [(drctn(b, a), dist(b, a), b) for b in asteroids if b != a])
               for a in asteroids]
  aligned = sorted([(a, len({d for d, _, _ in dists}))
                    for a, dists in distances],
                   key=lambda x: x[1], reverse=True)
  part1 = aligned[0][1]
  # clear enough but annoying
  best = aligned[0][0]
  from_best = dict(distances)[best]
  grouped = [(d, list(g)) for d, g in
             itt.groupby(sorted(from_best, key=lambda x: x[0]),
                         key=lambda x: x[0])]
  by_angle = [list(sorted(g, key=lambda g: abs(g[1][0])+abs(g[1][1])))
              for _, g in sorted(grouped, key=lambda x: angle(x[0]))]
  as_ds = [[(angle(di), abs(ds[0])+abs(ds[1]), p) for di, ds, p in g]
           for g in by_angle]
  longest = max(len(ps) for ps in as_ds)
  as_ds_pad = [ps + ([None] * (longest - len(ps))) for ps in as_ds]
  ordered = [p for ps in zip(*as_ds_pad) for p in ps if p != None]
  twohx, twohy = ordered[199][2]
  part2 = twohx * 100 + twohy
  return part1, part2


def day11(lines):
  """Solution to https://adventofcode.com/2019/day/11."""
  pass


def day12(lines):
  """Solution to https://adventofcode.com/2019/day/12."""
  pass


def day13(lines):
  """Solution to https://adventofcode.com/2019/day/13."""
  pass


def day14(lines):
  """Solution to https://adventofcode.com/2019/day/14."""
  pass


def day15(lines):
  """Solution to https://adventofcode.com/2019/day/15."""
  pass


def day16(lines):
  """Solution to https://adventofcode.com/2019/day/16."""
  pass


def day17(lines):
  """Solution to https://adventofcode.com/2019/day/17."""
  pass


def day18(lines):
  """Solution to https://adventofcode.com/2019/day/18."""
  pass


def day19(lines):
  """Solution to https://adventofcode.com/2019/day/19."""
  pass


def day20(lines):
  """Solution to https://adventofcode.com/2019/day/20."""
  pass


def day21(lines):
  """Solution to https://adventofcode.com/2019/day/21."""
  pass


def day22(lines):
  """Solution to https://adventofcode.com/2019/day/22."""
  pass


def day23(lines):
  """Solution to https://adventofcode.com/2019/day/23."""
  pass


def day24(lines):
  """Solution to https://adventofcode.com/2019/day/24."""
  pass


def day25(lines):
  """Solution to https://adventofcode.com/2019/day/25."""
  pass
