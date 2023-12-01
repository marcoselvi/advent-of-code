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
    return sum([digits(line) for line in lines])
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
    return sum([digits(line) for line in lines])
  testlines = get_lines('2023/day1.calibration.test2.txt')
  lines = get_lines('2023/day1.calibration.txt')
  return add_up(testlines), add_up(lines)


if __name__ == '__main__':
  print('Day 1 solutions:', day1_p1(), day1_p2())
