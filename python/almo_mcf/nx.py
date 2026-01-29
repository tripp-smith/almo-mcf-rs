"""NetworkX adapter for almo-mcf.

This module provides the public functions expected by the README specification.
The actual solver is expected to be provided by the Rust extension module when
available.
"""
from __future__ import annotations

from typing import Dict, Tuple

import math

import numpy as np


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
    if G.is_multigraph():
        raise ValueError("MultiDiGraph is not supported yet.")
    nodes = list(G.nodes())
    index = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)
    demand = [0] * n
    for node, idx in index.items():
        node_demand = G.nodes[node].get("demand", 0)
        if not isinstance(node_demand, (int, float)) or not math.isfinite(node_demand):
            raise ValueError("Node demand must be a finite number.")
        demand[idx] = int(node_demand)
    tails = []
    heads = []
    lower = []
    upper = []
    cost = []
    edges = list(G.edges(data=True))
    for u, v, data in edges:
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
        upper.append(int(capacity))
        edge_cost = data.get("weight", data.get("cost", 0))
        if not isinstance(edge_cost, (int, float)) or not math.isfinite(edge_cost):
            raise ValueError("Edge cost must be a finite number.")
        cost.append(int(edge_cost))
    return tails, heads, lower, upper, cost, demand, index, edges


def min_cost_flow(G) -> Dict:
    """Return a min-cost flow dict in NetworkX format.

    Args:
        G: NetworkX DiGraph with demand and capacity attributes.
    """
    core = _load_core()
    tails, heads, lower, upper, cost, demand, index, edges = _graph_to_arrays(G)
    flow = core.min_cost_flow_edges(
        len(index),
        np.asarray(tails, dtype=np.int64),
        np.asarray(heads, dtype=np.int64),
        np.asarray(lower, dtype=np.int64),
        np.asarray(upper, dtype=np.int64),
        np.asarray(cost, dtype=np.int64),
        np.asarray(demand, dtype=np.int64),
    )
    flow_dict: Dict = {node: {} for node in G.nodes()}
    for (u, v, _data), f in zip(edges, flow.tolist()):
        flow_dict[u][v] = int(f)
    return flow_dict


def min_cost_flow_cost(G, flow_dict: Dict) -> int:
    """Compute the total cost for a flow dict in NetworkX format."""
    total = 0
    for u, v, data in G.edges(data=True):
        flow = flow_dict[u][v]
        edge_cost = data.get("weight", data.get("cost", 0))
        total += int(flow) * int(edge_cost)
    return total
