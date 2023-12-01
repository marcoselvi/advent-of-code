load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary")

py_library(
    name = "utils",
    srcs = ["utils.py"],
)

py_library(
    name = "solve_2015",
    srcs = ["solve_2015.py"],
    data = [
        "data_2015/day1.txt",
        "data_2015/day10.txt",
        "data_2015/day11.txt",
        "data_2015/day12.txt",
        "data_2015/day13.txt",
        "data_2015/day14.txt",
        "data_2015/day15.txt",
        "data_2015/day16.txt",
        "data_2015/day17.txt",
        "data_2015/day18.txt",
        "data_2015/day19.txt",
        "data_2015/day2.txt",
        "data_2015/day20.txt",
        "data_2015/day21.txt",
        "data_2015/day22.txt",
        "data_2015/day23.txt",
        "data_2015/day24.txt",
        "data_2015/day25.txt",
        "data_2015/day3.txt",
        "data_2015/day4.txt",
        "data_2015/day5.txt",
        "data_2015/day6.txt",
        "data_2015/day7.txt",
        "data_2015/day8.txt",
        "data_2015/day9.txt",
    ],
    deps = [":utils"],
)

pytype_strict_binary(
    name = "2015",
    srcs = ["run_2015.py"],
    main = "run_2015.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":solve_2015",
        ":utils",
        "//third_party/py/absl:app",
    ],
)

py_test(
    name = "test_2015",
    srcs = ["test_2015.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":solve_2015",
        ":utils",
        "//testing/pybase",
    ],
)

py_library(
    name = "solve_2019",
    srcs = ["solve_2019.py"],
    data = [
        "data_2019/day1.txt",
        "data_2019/day10.txt",
        "data_2019/day11.txt",
        "data_2019/day12.txt",
        "data_2019/day13.txt",
        "data_2019/day14.txt",
        "data_2019/day15.txt",
        "data_2019/day16.txt",
        "data_2019/day17.txt",
        "data_2019/day18.txt",
        "data_2019/day19.txt",
        "data_2019/day2.txt",
        "data_2019/day20.txt",
        "data_2019/day21.txt",
        "data_2019/day22.txt",
        "data_2019/day23.txt",
        "data_2019/day24.txt",
        "data_2019/day25.txt",
        "data_2019/day3.txt",
        "data_2019/day4.txt",
        "data_2019/day5.txt",
        "data_2019/day6.txt",
        "data_2019/day7.txt",
        "data_2019/day8.txt",
        "data_2019/day9.txt",
    ],
    deps = [":utils"],
)

pytype_strict_binary(
    name = "2019",
    srcs = ["run_2019.py"],
    main = "run_2019.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":solve_2019",
        ":utils",
        "//third_party/py/absl:app",
    ],
)

py_test(
    name = "test_2019",
    srcs = ["test_2019.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":solve_2019",
        ":utils",
        "//testing/pybase",
    ],
)

py_library(
    name = "solve_2020",
    srcs = ["solve_2020.py"],
    data = [
        "data_2020/day1.txt",
        "data_2020/day10.txt",
        "data_2020/day11.txt",
        "data_2020/day12.txt",
        "data_2020/day13.txt",
        "data_2020/day14.txt",
        "data_2020/day15.txt",
        "data_2020/day16.txt",
        "data_2020/day17.txt",
        "data_2020/day18.txt",
        "data_2020/day19.txt",
        "data_2020/day2.txt",
        "data_2020/day20.txt",
        "data_2020/day21.txt",
        "data_2020/day22.txt",
        "data_2020/day23.txt",
        "data_2020/day24.txt",
        "data_2020/day25.txt",
        "data_2020/day3.txt",
        "data_2020/day4.txt",
        "data_2020/day5.txt",
        "data_2020/day6.txt",
        "data_2020/day7.txt",
        "data_2020/day8.txt",
        "data_2020/day9.txt",
    ],
    deps = [
        ":utils",
        "//third_party/py/numpy",
    ],
)

pytype_strict_binary(
    name = "2020",
    srcs = ["run_2020.py"],
    main = "run_2020.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":solve_2020",
        ":utils",
        "//third_party/py/absl:app",
    ],
)

py_test(
    name = "test_2020",
    srcs = ["test_2020.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":solve_2020",
        ":utils",
        "//testing/pybase",
    ],
)
