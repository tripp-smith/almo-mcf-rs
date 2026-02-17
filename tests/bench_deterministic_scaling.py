import math
import time


def fit_exponent(ns, ts):
    xs = [math.log(n) for n in ns]
    ys = [math.log(max(t, 1e-9)) for t in ts]
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    num = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    den = sum((x - xbar) ** 2 for x in xs)
    return num / den if den else 0.0


def test_time_validator_smoke():
    ns = [100, 200, 400, 800]
    ts = [n * math.log(n) for n in ns]
    exponent = fit_exponent(ns, ts)
    assert exponent <= 1.01 + 0.2
