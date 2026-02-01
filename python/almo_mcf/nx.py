"""NetworkX adapter for almo-mcf.

This module provides the public functions expected by the README specification.
The actual solver is expected to be provided by the Rust extension module when
available.
"""
from __future__ import annotations

from typing import Dict, Tuple

import math

import numpy as np

from .typing import FlowDict, MultiFlowDict

def _load_core():
    try:
        from . import _core  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised when extension is absent
        raise RuntimeError(
            "almo-mcf core extension is not built. "
            "Build the Rust extension via maturin to use the solver."
        ) from exc
    return _core


def _graph_to_arrays(G) -> Tuple[list, list, list, list, list, list, Dict, list]:
    if not G.is_directed():
        raise ValueError("Only directed graphs are supported.")
    nodes = list(G.nodes())
    index = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)
    demand = [0] * n
    for node, idx in index.items():
        node_demand = G.nodes[node].get("demand", 0)
        if not isinstance(node_demand, (int, float)) or not math.isfinite(node_demand):
            raise ValueError("Node demand must be a finite number.")
        demand[idx] = int(node_demand)
    if sum(demand) != 0:
        raise ValueError("Sum of node demands must be zero.")
    tails = []
    heads = []
    lower = []
    upper = []
    cost = []
    if G.is_multigraph():
        edges = list(G.edges(keys=True, data=True))
    else:
        edges = list(G.edges(data=True))
    for edge in edges:
        if G.is_multigraph():
            u, v, _key, data = edge
        else:
            u, v, data = edge
        tails.append(index[u])
        heads.append(index[v])
        lower_capacity = data.get("lower_capacity", 0)
        if not isinstance(lower_capacity, (int, float)) or not math.isfinite(lower_capacity):
            raise ValueError("lower_capacity must be a finite number.")
        lower.append(int(lower_capacity))
        if "capacity" not in data:
            raise ValueError("Each edge must specify a finite capacity.")
        capacity = data["capacity"]
        if not isinstance(capacity, (int, float)) or not math.isfinite(capacity):
            raise ValueError("Each edge must specify a finite capacity.")
        if capacity < lower_capacity:
            raise ValueError("capacity must be >= lower_capacity.")
        upper.append(int(capacity))
        edge_cost = data.get("weight", data.get("cost", 0))
        if not isinstance(edge_cost, (int, float)) or not math.isfinite(edge_cost):
            raise ValueError("Edge cost must be a finite number.")
        cost.append(int(edge_cost))
    return tails, heads, lower, upper, cost, demand, index, edges


def _gcd_of(values: list[int]) -> int:
    gcd_value = 0
    for value in values:
        if value != 0:
            gcd_value = math.gcd(gcd_value, abs(value))
    return gcd_value


def _scale_problem(
    lower: list[int],
    upper: list[int],
    cost: list[int],
    demand: list[int],
) -> tuple[list[int], list[int], list[int], list[int], int]:
    edge_count = max(1, len(cost))
    bound = edge_count**3
    max_capacity = max([abs(v) for v in lower + upper + demand] or [0])
    max_cost = max([abs(v) for v in cost] or [0])

    capacity_scale = 1
    while max_capacity > bound:
        gcd_value = _gcd_of(lower + upper + demand)
        if gcd_value <= 1:
            break
        lower = [v // gcd_value for v in lower]
        upper = [v // gcd_value for v in upper]
        demand = [v // gcd_value for v in demand]
        capacity_scale *= gcd_value
        max_capacity = max([abs(v) for v in lower + upper + demand] or [0])

    while max_cost > bound:
        gcd_value = _gcd_of(cost)
        if gcd_value <= 1:
            break
        cost = [v // gcd_value for v in cost]
        max_cost = max([abs(v) for v in cost] or [0])

    return lower, upper, cost, demand, capacity_scale


def min_cost_flow(
    G,
    *,
    use_ipm: bool | None = None,
    strategy: str | None = None,
    rebuild_every: int | None = None,
    max_iters: int | None = None,
    tolerance: float | None = None,
    seed: int | None = None,
    threads: int | None = None,
    alpha: float | None = None,
    approx_factor: float | None = None,
    return_stats: bool = False,
) -> FlowDict | MultiFlowDict | tuple[FlowDict | MultiFlowDict, dict | None]:
    """Return a min-cost flow dict in NetworkX format.

    Args:
        G: NetworkX DiGraph with demand and capacity attributes.
        use_ipm: Force enabling or disabling the IPM solver path.
        strategy: Optional solver strategy ("full_dynamic" or "periodic_rebuild").
        rebuild_every: Rebuild cadence for periodic rebuild strategy.
        max_iters: Maximum IPM iterations.
        tolerance: Convergence tolerance.
        seed: Random seed for IPM oracles.
        threads: Number of threads for solver execution.
        alpha: Optional override for the IPM barrier scaling constant.
        approx_factor: Approximation factor for the min-ratio cycle oracle.
        return_stats: When True, return (flow_dict, ipm_stats).
    """
    core = _load_core()
    tails, heads, lower, upper, cost, demand, index, edges = _graph_to_arrays(G)
    lower, upper, cost, demand, capacity_scale = _scale_problem(lower, upper, cost, demand)
    if hasattr(core, "min_cost_flow_edges_with_options"):
        flow, stats = core.min_cost_flow_edges_with_options(
            len(index),
            np.asarray(tails, dtype=np.int64),
            np.asarray(heads, dtype=np.int64),
            np.asarray(lower, dtype=np.int64),
            np.asarray(upper, dtype=np.int64),
            np.asarray(cost, dtype=np.int64),
            np.asarray(demand, dtype=np.int64),
            strategy=strategy,
            rebuild_every=rebuild_every,
            max_iters=max_iters,
            tolerance=tolerance,
            seed=seed,
            threads=threads,
            alpha=alpha,
            use_ipm=use_ipm,
            approx_factor=approx_factor,
        )
    else:
        flow = core.min_cost_flow_edges(
            len(index),
            np.asarray(tails, dtype=np.int64),
            np.asarray(heads, dtype=np.int64),
            np.asarray(lower, dtype=np.int64),
            np.asarray(upper, dtype=np.int64),
            np.asarray(cost, dtype=np.int64),
            np.asarray(demand, dtype=np.int64),
        )
        stats = None
    flow_values = np.asarray(flow, dtype=np.int64)
    if capacity_scale != 1:
        flow_values = flow_values * capacity_scale
    if G.is_multigraph():
        flow_dict: MultiFlowDict = {node: {} for node in G.nodes()}
        for (u, v, key, _data), f in zip(edges, flow_values.tolist()):
            flow_dict.setdefault(u, {}).setdefault(v, {})[key] = int(f)
    else:
        flow_dict = {node: {} for node in G.nodes()}
        for (u, v, _data), f in zip(edges, flow_values.tolist()):
            flow_dict[u][v] = int(f)
    if return_stats:
        return flow_dict, stats
    return flow_dict


def min_cost_flow_cost(G, flow_dict: FlowDict | MultiFlowDict) -> int:
    """Compute the total cost for a flow dict in NetworkX format."""
    total = 0
    if G.is_multigraph():
        edges = G.edges(keys=True, data=True)
    else:
        edges = G.edges(data=True)
    for edge in edges:
        if G.is_multigraph():
            u, v, key, data = edge
            flow = flow_dict.get(u, {}).get(v, {}).get(key, 0)
        else:
            u, v, data = edge
            flow = flow_dict.get(u, {}).get(v, 0)
        edge_cost = data.get("weight", data.get("cost", 0))
        total += int(flow) * int(edge_cost)
    return total
