"""Pure-Python fallback for the low-level array API.

This module is only imported when the compiled Rust extension is unavailable.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from ._version import __version__

__all__ = [
    "min_cost_flow_edges",
    "min_cost_flow_edges_with_options",
    "run_ipm_edges",
    "__version__",
]


def _as_int64_array(values: Iterable[int], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return array


def min_cost_flow_edges(
    n: int,
    tail: Iterable[int],
    head: Iterable[int],
    lower: Iterable[int],
    upper: Iterable[int],
    cost: Iterable[int],
    demand: Iterable[int],
) -> np.ndarray:
    """Compute the min-cost flow using the low-level array-based API.

    This fallback uses NetworkX's min_cost_flow implementation when the Rust
    extension is not installed.
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    tail_arr = _as_int64_array(tail, "tail")
    head_arr = _as_int64_array(head, "head")
    lower_arr = _as_int64_array(lower, "lower")
    upper_arr = _as_int64_array(upper, "upper")
    cost_arr = _as_int64_array(cost, "cost")
    demand_arr = _as_int64_array(demand, "demand")

    edge_count = len(tail_arr)
    if len(head_arr) != edge_count:
        raise ValueError("tail and head arrays must match length")
    if len(lower_arr) != edge_count or len(upper_arr) != edge_count or len(cost_arr) != edge_count:
        raise ValueError("edge attribute arrays must match tail/head length")
    if len(demand_arr) != n:
        raise ValueError("demand length does not match n")

    if np.any(tail_arr < 0) or np.any(tail_arr >= n):
        raise ValueError("tail index out of range")
    if np.any(head_arr < 0) or np.any(head_arr >= n):
        raise ValueError("head index out of range")
    if np.any(lower_arr > upper_arr):
        raise ValueError("lower capacity exceeds upper capacity")

    import networkx as nx

    demand_adjusted = demand_arr.astype(np.int64, copy=True)
    graph = nx.DiGraph()
    for node in range(n):
        graph.add_node(node, demand=int(demand_adjusted[node]))

    for idx in range(edge_count):
        lower_val = int(lower_arr[idx])
        upper_val = int(upper_arr[idx])
        tail_idx = int(tail_arr[idx])
        head_idx = int(head_arr[idx])
        cost_val = int(cost_arr[idx])
        capacity = upper_val - lower_val
        if capacity < 0:
            raise ValueError("lower capacity exceeds upper capacity")
        if lower_val:
            demand_adjusted[tail_idx] += lower_val
            demand_adjusted[head_idx] -= lower_val
        graph.add_edge(tail_idx, head_idx, capacity=capacity, weight=cost_val)

    for node in range(n):
        graph.nodes[node]["demand"] = int(demand_adjusted[node])

    flow_dict = nx.min_cost_flow(graph)
    flows = np.zeros(edge_count, dtype=np.int64)
    for idx in range(edge_count):
        tail_idx = int(tail_arr[idx])
        head_idx = int(head_arr[idx])
        flow_val = int(flow_dict.get(tail_idx, {}).get(head_idx, 0))
        flows[idx] = flow_val + int(lower_arr[idx])
    return flows


def min_cost_flow_edges_with_options(
    n: int,
    tail: Iterable[int],
    head: Iterable[int],
    lower: Iterable[int],
    upper: Iterable[int],
    cost: Iterable[int],
    demand: Iterable[int],
    *,
    strategy: str | None = None,
    rebuild_every: int | None = None,
    max_iters: int | None = None,
    tolerance: float | None = None,
    numerical_clamp_log: float | None = None,
    residual_min: float | None = None,
    barrier_alpha_min: float | None = None,
    barrier_alpha_max: float | None = None,
    barrier_clamp_max: float | None = None,
    gradient_clamp_max: float | None = None,
    log_numerical_clamping: bool | None = None,
    seed: int | None = None,
    deterministic_seed: int | None = None,
    threads: int | None = None,
    alpha: float | None = None,
    use_ipm: bool | None = None,
    approx_factor: float | None = None,
    deterministic: bool | None = None,
):
    """Compute the min-cost flow with optional solver tuning parameters.

    The fallback ignores tuning options and returns (flow, None).
    """
    _ = (
        strategy,
        rebuild_every,
        max_iters,
        tolerance,
        numerical_clamp_log,
        residual_min,
        barrier_alpha_min,
        barrier_alpha_max,
        barrier_clamp_max,
        gradient_clamp_max,
        log_numerical_clamping,
        seed,
        deterministic_seed,
        threads,
        alpha,
        use_ipm,
        approx_factor,
        deterministic,
    )
    flows = min_cost_flow_edges(n, tail, head, lower, upper, cost, demand)
    stats = {
        "solver_mode": "classic",
        "iterations": 0,
        "final_gap": 0.0,
        "termination": "classic",
        "deterministic_mode_used": bool(deterministic) if deterministic is not None else True,
        "seed_used": deterministic_seed if deterministic else seed,
        "numerical_clamping_occurred": False,
        "max_barrier_value": 0.0,
        "min_residual_seen": 0.0,
    }
    return flows, stats


def run_ipm_edges(
    n: int,
    tail: Iterable[int],
    head: Iterable[int],
    lower: Iterable[int],
    upper: Iterable[int],
    cost: Iterable[int],
    demand: Iterable[int],
    *,
    strategy: str | None = None,
    rebuild_every: int | None = None,
    max_iters: int | None = None,
    tolerance: float | None = None,
    numerical_clamp_log: float | None = None,
    residual_min: float | None = None,
    barrier_alpha_min: float | None = None,
    barrier_alpha_max: float | None = None,
    barrier_clamp_max: float | None = None,
    gradient_clamp_max: float | None = None,
    log_numerical_clamping: bool | None = None,
    seed: int | None = None,
    deterministic_seed: int | None = None,
    threads: int | None = None,
    alpha: float | None = None,
    use_ipm: bool | None = None,
    approx_factor: float | None = None,
    deterministic: bool | None = None,
):
    """Run the IPM solver directly and return (flow, stats) for debugging."""
    _ = (
        strategy,
        rebuild_every,
        max_iters,
        tolerance,
        numerical_clamp_log,
        residual_min,
        barrier_alpha_min,
        barrier_alpha_max,
        barrier_clamp_max,
        gradient_clamp_max,
        log_numerical_clamping,
        seed,
        deterministic_seed,
        threads,
        alpha,
        use_ipm,
        approx_factor,
        deterministic,
    )
    flows = min_cost_flow_edges(n, tail, head, lower, upper, cost, demand).astype(float)
    stats = {
        "solver_mode": "ipm",
        "iterations": 0,
        "final_gap": 0.0,
        "termination": "fallback",
        "deterministic_mode_used": bool(deterministic) if deterministic is not None else True,
        "seed_used": deterministic_seed if deterministic else seed,
        "numerical_clamping_occurred": False,
        "max_barrier_value": 0.0,
        "min_residual_seen": 0.0,
    }
    return flows, stats
