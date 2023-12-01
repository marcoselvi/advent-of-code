"""Utils for Advent of Code solutions."""

import functools as fnt
import itertools
import os


def raise_(err, msg=''):
  raise err(msg)


def daypath(year, day):
  wd = os.getcwd()
  if wd.endswith('google3'):
    return f'experimental/users/marcoselvi/advent/data_{year}/{day}.txt'
  else:
    return f'data_{year}/{day}.txt'


def runday(dayfn, year, day):
  with open(daypath(year, day), 'r') as f:
    return dayfn([l.strip() for l in f.readlines()])


def conspairs(xs):
  return list(zip(xs[:-1], xs[1:]))


def identity(x):
  return x


def fst(xs):
  return xs[0]


def snd(xs):
  return xs[1]


def compose(f, g):
  def h(*args):
    return f(g(*args))
  return h


def group_sndbyfst(xs):
  return [(x, [y for _, y in group])
          for x, group in itertools.groupby(sorted(xs, key=fst), key=fst)]


def map_accum(f, xs, z):
  ys = []
  for x in xs:
    y, z = f(x, z)
    ys.append(y)
  return ys, z


def memoise(f):
  memo = {}
  def memoised_f(*args):
    if args in memo:
      return memo[args]
    y = f(*args)
    memo[args] = y
    return y
  return memoised_f


def setmemoise(f):
  memo = {}
  def memoised_f(*sets):
    key = tuple(frozenset(s) for s in sets)
    if key in memo:
      return memo[key]
    memo[key] = f(*key)
    return memo[key]
  return memoised_f


def dictmemoise(f):
  memo = {}
  def memoised_f(d):
    key = tuple(d.items())
    if key in memo:
      return memo[key]
    memo[key] = f(d)
    return memo[key]
  return memoised_f


def npmemoise(f):
  memo = {}
  def memoised_f(array):
    key = tuple(array.flatten())
    if key in memo:
      return memo[key]
    memo[key] = f(array)
    return memo[key]
  return memoised_f

