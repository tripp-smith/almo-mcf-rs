#!/usr/bin/env python3
"""Quick scaling benchmark for almost-linear behavior."""
from __future__ import annotations
import math
import random
import statistics
import time
import numpy as np

import almo_mcf as am


def make_instance(n: int, m: int, seed: int):
    rng = random.Random(seed)
    tails, heads, lower, upper, cost = [], [], [], [], []
    for _ in range(m):
        u = rng.randrange(n)
        v = rng.randrange(n)
        while v == u:
            v = rng.randrange(n)
        tails.append(u)
        heads.append(v)
        lower.append(0)
        upper.append(rng.randint(1, 1 << 10))
        cost.append(rng.randint(1, 1 << 8))
    demands = [0] * n
    demands[0] = -50
    demands[1] = 50
    return tails, heads, lower, upper, cost, demands


def run_once(n: int, m: int, seed: int) -> float:
    tails, heads, lower, upper, cost, demands = make_instance(n, m, seed)
    t0 = time.perf_counter()
    _ = am.solve_min_cost_flow(
        tails=tails,
        heads=heads,
        lower=lower,
        upper=upper,
        cost=cost,
        demands=demands,
        options={"solver_mode": "ipm", "deterministic": True},
    )
    return time.perf_counter() - t0


def main() -> None:
    sizes = [(n, 10 * n) for n in [100, 300, 600, 1000, 2000, 3000]]
    xs, ys = [], []
    for n, m in sizes:
        samples = [run_once(n, m, 100 + i) for i in range(5)]
        mean = statistics.mean(samples)
        std = statistics.pstdev(samples)
        assert std <= 0.05 * mean + 1e-9, f"variance too high at n={n}: std={std}, mean={mean}"
        scale = m * math.log(max(m, 2)) * math.log((1 << 10) + (1 << 30))
        xs.append(math.log(scale))
        ys.append(math.log(mean))
        print(f"n={n} m={m} mean={mean:.4f}s std={std:.4f}s")

    slope, _ = np.polyfit(xs, ys, 1)
    print(f"fitted exponent={slope:.4f}")
    assert slope <= 1.05, f"scaling exponent too high: {slope}"


if __name__ == "__main__":
    main()
