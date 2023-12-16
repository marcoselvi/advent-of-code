"""Solutions to Advent of Code 2015."""

import fractions
import functools as fnt
import hashlib
import itertools as itt
import json
import math
import re

from .. import utils


def day1(lines):
  """Solution to https://adventofcode.com/2015/day/1."""
  def move(floor, inst):
    return floor + 1 if inst == '(' else floor - 1
  def find_index(instructions, floor, i):
    if floor < 0:
      return i
    return find_index(instructions[1:], move(floor, instructions[0]), i+1)
  instructions = lines[0]
  part1 = fnt.reduce(move, instructions, 0)
  part2 = find_index(instructions, 0, 0)
  return part1, part2


def day2(lines):
  """Solution to https://adventofcode.com/2015/day/2."""
  boxes = [list(map(int, l.split('x'))) for l in lines]
  sides = [(a*b, a*c, b*c) for a, b, c in boxes]
  part1 = sum([2*s1 + 2*s2 + 2*s3 + min([s1, s2, s3]) for s1, s2, s3 in sides])
  part2 = sum([a*b*c + 2*sum(sorted([a, b, c])[:2]) for a, b, c in boxes])
  return part1, part2


def day3(lines):
  """Solution to https://adventofcode.com/2015/day/3."""
  def split(insts):
    return [insts[i:i+2] for i in range(0, len(insts), 2)]
  def updatepos(pos, inst):
    x, y = pos
    return (x+(1 if inst == '>' else -1 if inst == '<' else 0),
            y+(1 if inst == '^' else -1 if inst == 'v' else 0))
  def move1(state, inst):
    visited, pos = state
    newpos = updatepos(pos, inst)
    return visited + [newpos], newpos
  def move2(state, inst):
    visited, pos1, pos2 = state
    inst1, inst2 = inst
    newpos1 = updatepos(pos1, inst1)
    newpos2 = updatepos(pos2, inst2)
    return visited + [newpos1, newpos2], newpos1, newpos2
  instructions = lines[0]
  visited1, _ = fnt.reduce(move1, instructions, ([], (0, 0)))
  part1 = len(set(visited1))
  visited2, _, _ = fnt.reduce(move2, split(instructions), ([], (0, 0), (0, 0)))
  part2 = len(set(visited2))
  return part1, part2


def day4(lines):
  """Solution to https://adventofcode.com/2015/day/3."""
  inp = lines[0]
  def md5hash(n):
    return hashlib.md5((inp + str(n)).encode()).hexdigest()
  def prefix(pref):
    l = len(pref)
    x = 0
    while md5hash(x)[:l] != pref:
      x += 1
    return x
  part1 = prefix('00000')
  part2 = prefix('000000')
  return part1, part2


def day5(lines):
  """Solution to https://adventofcode.com/2015/day/5."""
  def nice1(l):
    return (len([c for c in l if c in 'aeiou']) >= 3 and
            any([x == y for x, y in utils.conspairs(l)]) and
            all([s not in l for s in ['ab', 'cd', 'pq', 'xy']]))
  def nice2(l):
    return (any([l[i] == l[i+2] for i in range(len(l)-2)]) and
            any([l[i:i+2] == l[j:j+2] for i in range(len(l)-1)  # pylint: disable=g-complex-comprehension
                 for j in range(i+2, len(l)-1)]))
  part1 = len([l for l in lines if nice1(l)])
  part2 = len([l for l in lines if nice2(l)])
  return part1, part2


def day6(lines):
  """Solution to https://adventofcode.com/2015/day/6."""
  def freshlights():
    return {(i, j): 0 for i in range(1000) for j in range(1000)}  # pylint: disable=g-complex-comprehension
  def linesplit(line):
    def splitran(ran):
      return list(map(int, ran.split(',')))
    pieces = line.split(' ')
    return ' '.join(pieces[:-3]), splitran(pieces[-3]), splitran(pieces[-1])
  def parse(commands, line):
    comm, ran1, ran2 = linesplit(line)
    return commands[comm], ran1, ran2
  def part1parse():
    def on(_):
      return 1
    def off(_):
      return 0
    def toggle(light):
      return int(not light)
    return {'turn on': on, 'turn off': off, 'toggle': toggle}
  def part2parse():
    def on(light):
      return light+1
    def off(light):
      return light-1 if light > 0 else 0
    def toggle(light):
      return light+2
    return {'turn on': on, 'turn off': off, 'toggle': toggle}
  def switch(parser, state, line):
    command, (istart, jstart), (iend, jend) = parser(line)
    for i in range(istart, iend+1):
      for j in range(jstart, jend+1):
        state[(i, j)] = command(state[(i, j)])
    return state
  def sumvals(d):
    return sum(d.values())
  part1 = sumvals(fnt.reduce(
      fnt.partial(switch, fnt.partial(parse, part1parse())),
      lines, freshlights()))
  part2 = sumvals(fnt.reduce(
      fnt.partial(switch, fnt.partial(parse, part2parse())),
      lines, freshlights()))
  return part1, part2


def day7(lines):
  """Solution to https://adventofcode.com/2015/day/7."""
  def mask(n):
    return n & 0xFFFF
  gates = {'OR': lambda x, y: mask(x | y), 'AND': lambda x, y: mask(x & y),
           'NOT': lambda x: mask(~x),
           'LSHIFT': lambda x, y: mask(x << y),
           'RSHIFT': lambda x, y: mask(x >> y)}
  def getkey(line):
    return line.split(' -> ')[-1]
  def getsource(line):
    return line.split(' -> ')[0].split(' ')
  def solver(insts):
    @utils.memoise
    def resolve(key):
      if key not in insts:
        return int(key)
      if len(insts[key]) == 1:
        return resolve(insts[key][0])
      if len(insts[key]) == 2:
        gate, val = insts[key]
        return gates[gate](resolve(val))
      val1, gate, val2 = insts[key]
      return gates[gate](resolve(val1), resolve(val2))
    return resolve
  instructions = {getkey(line): getsource(line) for line in lines}
  part1 = solver(instructions)('a')
  part2 = solver({**instructions, **{'b': [str(part1)]}})('a')
  return part1, part2


def day8(lines):
  """Solution to https://adventofcode.com/2015/day/8."""
  def countchars(l, count):
    if not l:
      return count
    newline = (l[4:] if l[0] == '\\' and l[1] == 'x' else
               l[2:] if l[0] == '\\' else
               l[1:])
    return countchars(newline, count+1)
  def extendstring(s, ch):
    return s + ('\\\\' if ch == '\\' else
                '\\\"' if ch == '\"' else
                ch)
  allchars = sum(len(l) for l in lines)
  parsedchars = sum(countchars(l[1:-1], 0) for l in lines)
  part1 = allchars - parsedchars
  extendedstrings = ['\"' + fnt.reduce(extendstring, l, '') + '\"'
                     for l in lines]
  part2 = sum(len(l) for l in extendedstrings) - allchars
  return part1, part2


def day9(lines):
  """Solution to https://adventofcode.com/2015/day/9."""
  def parse(g, line):
    places, d = line.split(' = ')
    fro, to = places.split(' to ')
    g[fro] = g.get(fro, []) + [(to, int(d))]
    g[to] = g.get(to, []) + [(fro, int(d))]
    return g
  def path(filt, graph, loc, tovisit, dist, fallback):
    if not tovisit:
      return dist
    if tovisit & {to for to, _ in graph[loc]} == set():
      return fallback
    return filt([path(filt, graph, to, tovisit-{to}, dist+d, fallback)
                 for to, d in graph[loc] if to in tovisit])
  graph = fnt.reduce(parse, lines, {})
  part1 = min([path(min, graph, loc, graph.keys()-{loc}, 0, math.inf)
               for loc in graph])
  part2 = max([path(max, graph, loc, graph.keys()-{loc}, 0, 0)
               for loc in graph])
  return part1, part2


def day10(lines):
  """Solution to https://adventofcode.com/2015/day/10."""
  def update(digs, _):
    return ''.join([str(len(list(g))) + str(k) for k, g in itt.groupby(digs)])
  digits = lines[0]
  after40 = fnt.reduce(update, range(40), digits)
  part1 = len(after40)
  part2 = len(fnt.reduce(update, range(10), after40))
  return part1, part2


def day11(lines):
  """Solution to https://adventofcode.com/2015/day/11."""
  alph = 'abcdefghijklmnopqrstuvwxyz'
  consecutives = {alph[i:i+3] for i in range(len(alph)-2)}
  def increment(p):
    def newi(i):
      return (i+1) % len(alph)
    def update(i, r):
      return (newi(i) if r else i, 1 if r and newi(i) == 0 else 0)
    idxs = [alph.index(ch) for ch in p]
    newidxs = reversed(utils.map_accum(update, reversed(idxs), 1)[0])
    return ''.join([alph[i] for i in newidxs])
  def valid1(p):
    return any(cons in p for cons in consecutives)
  def valid2(p):
    return all(ch not in p for ch in {'i', 'l', 'o'})
  def valid3(p):
    def pairs(n_j, i_pair):
      (n, j), (i, (x, y)) = n_j, i_pair
      return (n+1 if x == y and i != j+1 else n, i if x == y else j)
    return fnt.reduce(pairs, enumerate(utils.conspairs(p)), (0, -2))[0] >= 2
  def valid(p):
    return valid1(p) and valid2(p) and valid3(p)
  def nextpassword(oldp):
    p = oldp
    while not valid(p):
      p = increment(p)
    return p
  part1 = nextpassword(lines[0])
  part2 = nextpassword(increment(part1))
  return part1, part2


def day12(lines):
  """Solution to https://adventofcode.com/2015/day/12."""
  def recurse(fn, d):
    return (sum([fn(v) for v in d.values()]) if isinstance(d, dict) else  # pylint: disable=g-long-ternary
            sum([fn(v) for v in d]) if isinstance(d, list) else
            d if isinstance(d, int) else
            0)
  def sumints(d):
    return recurse(sumints, d)
  def sumnonred(d):
    return (0 if isinstance(d, dict) and 'red' in d.values() else  # pylint: disable=g-long-ternary
            recurse(sumnonred, d))
  data = json.loads(lines[0])
  part1 = sumints(data)
  part2 = sumnonred(data)
  return part1, part2


def day13(lines):
  """Solution to https://adventofcode.com/2015/day/13."""
  def parse_line(line):
    m = re.match(r"(\w+) would (\w+) (\d+) happiness units by sitting next to (\w+)\.", line)
    return m.group(1), m.group(4), (-1 if m.group(2) == 'lose' else 1) * int(m.group(3))
  def happiness(table, guests):
    return sum(table[(ga, gb)] + table[(gb, ga)] for ga, gb in zip(guests, guests[1:] + (guests[0],)))
  def table_happiness(table):
    guests = {ga for ga, _ in table.keys()}
    return max(happiness(table, ring) for ring in itt.permutations(guests))
  table = {(ga, gb): happy for ga, gb, happy in map(parse_line, lines)}
  part1 = table_happiness(table)
  part2 = table_happiness(table | {('Me', ga): 0 for ga, _ in table.keys()} | {(ga, 'Me'): 0 for ga, _ in table.keys()})
  return part1, part2


def day14(lines):
  """Solution to https://adventofcode.com/2015/day/14."""
  T = 2503
  def parse_line(line):
    m = re.match(r"(\w+) can fly (\d+) km/s for (\d+) seconds, but then must rest for (\d+) seconds\.", line)
    return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
  def distance(name_speed_runtime_resttime):
    name, speed, runtime, resttime = name_speed_runtime_resttime
    d, c, r = 0, T, 'run'
    while c > 0:
      if r == 'run':
        d += speed * min(runtime, c)
        c -= runtime
      else:
        c -= resttime
      r = 'run' if r == 'rest' else 'rest'
    return name, d
  distances = sorted(map(utils.compose(distance, parse_line), lines), key=utils.snd)
  part1 = distances[-1][1]
  # def race(all_reindeers):
  part2 = None
  return part1, part2


def day15(lines):
  """Solution to https://adventofcode.com/2015/day/15."""
  pass


def day16(lines):
  """Solution to https://adventofcode.com/2015/day/16."""
  pass


def day17(lines):
  """Solution to https://adventofcode.com/2015/day/17."""
  pass


def day18(lines):
  """Solution to https://adventofcode.com/2015/day/18."""
  pass


def day19(lines):
  """Solution to https://adventofcode.com/2015/day/19."""
  pass


def day20(lines):
  """Solution to https://adventofcode.com/2015/day/20."""
  pass


def day21(lines):
  """Solution to https://adventofcode.com/2015/day/21."""
  pass


def day22(lines):
  """Solution to https://adventofcode.com/2015/day/22."""
  pass


def day23(lines):
  """Solution to https://adventofcode.com/2015/day/23."""
  pass


def day24(lines):
  """Solution to https://adventofcode.com/2015/day/24."""
  pass


def day25(lines):
  """Solution to https://adventofcode.com/2015/day/25."""
  pass
