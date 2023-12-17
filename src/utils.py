"""Utils for Advent of Code."""

import functools as fnt
import itertools
import os


bind = fnt.partial


def daypath(year, day):
  wd = os.getcwd()
  return f'data/y{year}/{day}.txt'


def runday(dayfn, year, day):
  with open(daypath(year, day), 'r') as f:
    return dayfn([l.strip() for l in f.readlines()])


def raise_(err, msg=''):
  raise err(msg)


def conspairs(xs):
  return list(zip(xs[:-1], xs[1:]))


def identity(x):
  return x


def fst(xs):
  return xs[0]


def snd(xs):
  return xs[1]


def join(xss):
  return [x for xs in xss for x in xs]


def transpose_strs(xs):
  return tuple(''.join(x) for x in zip(*xs))


def compose(f, g):
  def h(*args):
    return f(g(*args))
  return h


def minval_keyval(d):
  m, kk = None, None
  for k, v in d.items():
    m = v if not m else min(m, v)
    kk = k if min(m, v) == v else kk
  return kk, m


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
    if args not in memo:
      memo[args] = f(*args)
    return memo[args]
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

