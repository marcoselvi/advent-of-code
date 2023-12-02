import re


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
  testlines = get_lines('2023/day1.calibration.test1.txt')
  lines = get_lines('2023/day1.calibration.txt')
  return add_up(testlines), add_up(lines)

def day1_p2():
  def digits(line):
    pattern = '|'.join(WORD_TO_DIGIT_MAP.keys())
    numbers = [WORD_TO_DIGIT_MAP.get(maybe_word[0], maybe_word[1])
               for maybe_word in re.findall(rf'(?=({pattern}))|(\d)', line)]
    return int(numbers[0] + numbers[-1])
  def add_up(lines):
    return sum(digits(line) for line in lines)
  testlines = get_lines('2023/day1.calibration.test2.txt')
  lines = get_lines('2023/day1.calibration.txt')
  return add_up(testlines), add_up(lines)


def day2_p1():
  available = {'red': 12, 'green': 13, 'blue': 14}
  def is_possible(game):
    rounds = game.split(':')[-1].split(';')
    for r in rounds:
      for cubes in r.split(','):
        n, c = cubes.strip().split(' ')
        if available[c] < int(n):
          return False
    return True
  def possible_games(games):
    return filter(is_possible, games)
  def game_id(game):
    return int(game.split(':')[0].split(' ')[-1])
  def add_up(games):
    return sum(game_id(game) for game in games)
  testgames = get_lines('2023/day2.cubes.test1.txt')
  games = get_lines('2023/day2.cubes.txt')
  return add_up(possible_games(testgames)), add_up(possible_games(games))

def day2_p2():
  def power(game):
    rounds = [[p.strip().split(' ') for p in r.split(',')] for r in game.split(':')[-1].split(';')]
    reds = [int(n) for roun in rounds for n, c in roun if c == 'red']
    blues = [int(n) for roun in rounds for n, c in roun if c == 'blue']
    greens = [int(n) for roun in rounds for n, c in roun if c == 'green']
    return max(reds) * max(blues) * max(greens)
  testgames = get_lines('2023/day2.cubes.test1.txt')
  games = get_lines('2023/day2.cubes.txt')
  return sum(power(game) for game in testgames), sum(power(game) for game in games)



if __name__ == '__main__':
  print('Day 1 solutions:', day1_p1(), day1_p2())
  print('Day 2 solutions:', day2_p1(), day2_p2())
