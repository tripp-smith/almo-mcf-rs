"""Pure-Python fallback for the low-level array API.

This module is only imported when the compiled Rust extension is unavailable.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from ._version import __version__

__all__ = ["min_cost_flow_edges", "__version__"]


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
