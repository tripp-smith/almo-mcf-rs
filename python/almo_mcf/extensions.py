"""Extensions and reductions built on top of the core min-cost flow solver."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import math

import networkx as nx
import numpy as np

from .nx import min_cost_flow
from .typing import FlowDict, MultiFlowDict, Node

ConvexCost = Callable[[int], float]


def _marginal_costs_from_callable(cost_fn: ConvexCost, capacity: int) -> list[float]:
    costs = []
    previous = cost_fn(0)
    for flow in range(1, capacity + 1):
        current = cost_fn(flow)
        costs.append(current - previous)
        previous = current
    return costs


def _marginal_costs_from_sequence(costs: Sequence[float], capacity: int) -> list[float]:
    if len(costs) == capacity:
        return list(costs)
    if len(costs) == capacity + 1:
        return [costs[i] - costs[i - 1] for i in range(1, capacity + 1)]
    raise ValueError("Convex cost sequence must have length capacity or capacity+1.")


def _edge_marginals(
    data: Mapping[str, Any],
    capacity: int,
    objective: str,
    p: float,
    scale: float,
) -> list[float]:
    cost_spec = data.get("convex_cost")
    if callable(cost_spec):
        return _marginal_costs_from_callable(cost_spec, capacity)
    if isinstance(cost_spec, Sequence):
        return _marginal_costs_from_sequence(cost_spec, capacity)

    weight = float(data.get("weight", data.get("cost", 1.0)))
    if objective == "p_norm":
        def cost_fn(flow: int) -> float:
            return scale * weight * (flow**p)

        return _marginal_costs_from_callable(cost_fn, capacity)
    if objective == "entropy":
        def cost_fn(flow: int) -> float:
            if flow <= 0:
                return 0.0
            return scale * weight * flow * math.log(flow)

        return _marginal_costs_from_callable(cost_fn, capacity)
    if objective == "matrix_scaling":
        def cost_fn(flow: int) -> float:
            if flow <= 0:
                return 0.0
            return scale * weight * (flow * math.log(flow) - flow)

        return _marginal_costs_from_callable(cost_fn, capacity)

    raise ValueError(f"Unsupported convex objective: {objective}")


def min_cost_flow_convex(
    G: nx.DiGraph | nx.MultiDiGraph,
    *,
    objective: str = "p_norm",
    p: float = 2.0,
    scale: float = 1.0,
    max_units: int = 10_000,
) -> FlowDict | MultiFlowDict:
    """Solve min-cost flow with separable convex costs via unit expansion."""
    if not G.is_directed():
        raise ValueError("Only directed graphs are supported.")

    expanded = nx.MultiDiGraph()
    expanded.add_nodes_from(G.nodes())
    demand = {node: int(G.nodes[node].get("demand", 0)) for node in G.nodes()}

    edge_keys = list(G.edges(keys=True, data=True)) if G.is_multigraph() else list(G.edges(data=True))
    edge_map: dict[tuple[Node, Node, Any], list[tuple[Node, Node, Any]]] = {}

    for edge in edge_keys:
        if G.is_multigraph():
            u, v, key, data = edge
            edge_id = (u, v, key)
        else:
            u, v, data = edge
            edge_id = (u, v, None)

        lower = int(data.get("lower_capacity", 0))
        capacity = data.get("capacity")
        if capacity is None:
            raise ValueError("Each edge must specify a finite capacity.")
        if not isinstance(capacity, (int, float)) or not math.isfinite(capacity):
            raise ValueError("Each edge must specify a finite capacity.")
        upper = int(capacity)
        if lower > upper:
            raise ValueError("capacity must be >= lower_capacity.")
        residual = upper - lower
        if residual > max_units:
            raise ValueError("Convex expansion would exceed max_units.")

        if lower:
            demand[u] = demand.get(u, 0) + lower
            demand[v] = demand.get(v, 0) - lower

        if residual == 0:
            edge_map[edge_id] = []
            continue

        marginals = _edge_marginals(data, residual, objective, p, scale)
        if any(marginals[i] > marginals[i + 1] for i in range(len(marginals) - 1)):
            raise ValueError("Convex cost marginals must be non-decreasing.")

        edge_map[edge_id] = []
        for idx, marginal in enumerate(marginals):
            unit_key = f"{edge_id}-{idx}"
            expanded.add_edge(u, v, key=unit_key, capacity=1, weight=float(marginal))
            edge_map[edge_id].append((u, v, unit_key))

    for node, node_demand in demand.items():
        expanded.nodes[node]["demand"] = int(node_demand)

    flow = min_cost_flow(expanded)
    flow_dict: MultiFlowDict = {node: {} for node in G.nodes()}
    for edge_id, expanded_edges in edge_map.items():
        u, v, key = edge_id
        total_flow = 0
        for edge_u, edge_v, unit_key in expanded_edges:
            total_flow += flow.get(edge_u, {}).get(edge_v, {}).get(unit_key, 0)
        if key is None:
            flow_dict.setdefault(u, {})[v] = total_flow
        else:
            flow_dict.setdefault(u, {}).setdefault(v, {})[key] = total_flow
    return flow_dict


def max_flow_via_min_cost_circulation(
    G: nx.DiGraph,
    source: Node,
    sink: Node,
    *,
    capacity: str = "capacity",
) -> tuple[int, FlowDict]:
    """Compute max flow using a min-cost circulation reduction."""
    if not G.is_directed():
        raise ValueError("Only directed graphs are supported.")

    base = nx.DiGraph()
    base.add_nodes_from(G.nodes())
    total_capacity = 0

    for u, v, data in G.edges(data=True):
        cap = data.get(capacity)
        if cap is None:
            raise ValueError("Each edge must specify a finite capacity.")
        if not isinstance(cap, (int, float)) or not math.isfinite(cap):
            raise ValueError("Each edge must specify a finite capacity.")
        base.add_edge(u, v, capacity=int(cap), weight=0)
        if u == source:
            total_capacity += int(cap)

    def build_graph(flow_value: int) -> nx.DiGraph:
        circulation = base.copy()
        circulation.add_edge(sink, source, capacity=flow_value, weight=0)
        for node in circulation.nodes():
            circulation.nodes[node]["demand"] = 0
        circulation.nodes[source]["demand"] = -flow_value
        circulation.nodes[sink]["demand"] = flow_value
        return circulation

    low, high = 0, total_capacity
    best_flow = 0
    best_solution: FlowDict = {node: {} for node in G.nodes()}
    while low <= high:
        mid = (low + high) // 2
        graph = build_graph(mid)
        try:
            flow_dict = min_cost_flow(graph)
        except Exception:
            high = mid - 1
            continue
        best_flow = mid
        best_solution = {node: {} for node in G.nodes()}
        for u, v in G.edges():
            best_solution[u][v] = int(flow_dict.get(u, {}).get(v, 0))
        low = mid + 1

    return best_flow, best_solution


def bipartite_min_cost_matching(
    G: nx.Graph,
    left_nodes: Iterable[Node],
    right_nodes: Iterable[Node],
    *,
    weight: str = "weight",
) -> dict[Node, Node]:
    """Compute a minimum-cost maximum matching on a bipartite graph."""
    left_nodes = list(left_nodes)
    right_nodes = list(right_nodes)
    maximum = nx.algorithms.bipartite.maximum_matching(G, top_nodes=left_nodes)
    max_cardinality = len(maximum) // 2
    if max_cardinality == 0:
        return {}

    flow_graph = nx.DiGraph()
    flow_graph.add_nodes_from(G.nodes())
    flow_graph.add_node("__source__", demand=-max_cardinality)
    flow_graph.add_node("__sink__", demand=max_cardinality)

    for node in left_nodes:
        flow_graph.add_edge("__source__", node, capacity=1, weight=0)
    for node in right_nodes:
        flow_graph.add_edge(node, "__sink__", capacity=1, weight=0)
    for u, v, data in G.edges(data=True):
        if u in left_nodes and v in right_nodes:
            edge_cost = data.get(weight, 0)
            flow_graph.add_edge(u, v, capacity=1, weight=edge_cost)
        elif v in left_nodes and u in right_nodes:
            edge_cost = data.get(weight, 0)
            flow_graph.add_edge(v, u, capacity=1, weight=edge_cost)

    flow = min_cost_flow(flow_graph)
    matching: dict[Node, Node] = {}
    for u in left_nodes:
        for v, f in flow.get(u, {}).items():
            if v in right_nodes and f == 1:
                matching[u] = v
                matching[v] = u
    return matching


def find_negative_cycle(G: nx.DiGraph, *, weight: str = "weight") -> list[Node] | None:
    """Return a negative cycle if one exists."""
    nodes = list(G.nodes())
    if not nodes:
        return None
    dist = {node: 0.0 for node in nodes}
    pred: dict[Node, Node] = {}

    for i in range(len(nodes)):
        updated = False
        for u, v, data in G.edges(data=True):
            w = float(data.get(weight, 0.0))
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u
                updated = True
                if i == len(nodes) - 1:
                    cycle_node = v
                    for _ in range(len(nodes)):
                        cycle_node = pred.get(cycle_node, cycle_node)
                    cycle = [cycle_node]
                    current = pred.get(cycle_node)
                    while current is not None and current not in cycle:
                        cycle.append(current)
                        current = pred.get(current)
                    cycle.append(cycle_node)
                    cycle.reverse()
                    return cycle
        if not updated:
            break
    return None


def vertex_connectivity(G: nx.Graph) -> int:
    """Compute vertex connectivity using NetworkX."""
    return nx.node_connectivity(G)


def gomory_hu_tree(G: nx.Graph, *, capacity: str = "capacity") -> nx.Graph:
    """Compute a Gomory-Hu tree for an undirected graph."""
    return nx.gomory_hu_tree(G, capacity=capacity)


def sparsest_cut(
    G: nx.Graph,
    *,
    capacity: str = "capacity",
) -> tuple[float, tuple[set[Node], set[Node]]]:
    """Compute a sparsest cut by scanning all-pairs min-cuts."""
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return 0.0, (set(nodes), set())

    best_ratio = math.inf
    best_partition: tuple[set[Node], set[Node]] = (set(), set())

    for i, source in enumerate(nodes):
        for target in nodes[i + 1 :]:
            cut_value, (partition, complement) = nx.minimum_cut(G, source, target, capacity=capacity)
            smaller = min(len(partition), len(complement))
            if smaller == 0:
                continue
            ratio = float(cut_value) / float(smaller)
            if ratio < best_ratio:
                best_ratio = ratio
                best_partition = (set(partition), set(complement))

    if best_ratio is math.inf:
        best_ratio = 0.0
    return best_ratio, best_partition


def isotonic_regression_dag(
    G: nx.DiGraph,
    values: Mapping[Node, float],
    *,
    weights: Mapping[Node, float] | None = None,
    max_iters: int = 10_000,
    tol: float = 1e-9,
) -> dict[Node, float]:
    """Perform isotonic regression on a DAG with L2 objective."""
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph must be a directed acyclic graph.")

    nodes = list(G.nodes())
    index = {node: idx for idx, node in enumerate(nodes)}
    x = np.array([float(values[node]) for node in nodes], dtype=float)
    w = np.ones_like(x)
    if weights is not None:
        w = np.array([float(weights.get(node, 1.0)) for node in nodes], dtype=float)
        if np.any(w <= 0):
            raise ValueError("Weights must be positive.")

    for _ in range(max_iters):
        updated = False
        for u, v in G.edges():
            i = index[u]
            j = index[v]
            if x[i] > x[j] + tol:
                avg = (w[i] * x[i] + w[j] * x[j]) / (w[i] + w[j])
                x[i] = avg
                x[j] = avg
                updated = True
        if not updated:
            break

    return {node: float(x[index[node]]) for node in nodes}
