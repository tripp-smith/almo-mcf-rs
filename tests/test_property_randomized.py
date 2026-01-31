import random

import networkx as nx
from hypothesis import given, settings, strategies as st

from almo_mcf import min_cost_flow, min_cost_flow_cost

from .regression_seeds import SEEDS as REGRESSION_SEEDS


def _build_random_graph(seed: int, node_count: int | None = None) -> nx.DiGraph:
    rng = random.Random(seed)
    if node_count is None:
        node_count = rng.randint(5, 30)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(node_count))

    for idx in range(node_count):
        next_idx = (idx + 1) % node_count
        graph.add_edge(
            idx,
            next_idx,
            capacity=rng.randint(1, 8),
            weight=rng.randint(0, 10),
        )

    for u in range(node_count):
        for v in range(node_count):
            if u == v or graph.has_edge(u, v):
                continue
            if rng.random() < 0.15:
                graph.add_edge(
                    u,
                    v,
                    capacity=rng.randint(1, 8),
                    weight=rng.randint(0, 10),
                )

    demands = [0] * node_count
    for u, v, data in graph.edges(data=True):
        flow_value = rng.randint(0, min(2, data["capacity"]))
        demands[u] -= flow_value
        demands[v] += flow_value

    for node, demand in enumerate(demands):
        graph.nodes[node]["demand"] = demand

    return graph


def _assert_flow_feasible(graph: nx.DiGraph, flow) -> None:
    for u, v, data in graph.edges(data=True):
        assert 0 <= flow[u][v] <= data["capacity"]

    for node in graph.nodes():
        inflow = sum(flow[u][node] for u in graph.predecessors(node))
        outflow = sum(flow[node][v] for v in graph.successors(node))
        assert inflow - outflow == graph.nodes[node].get("demand", 0)


def test_random_graphs_match_networkx():
    cases = [
        (0, 5),
        (1, 10),
        (2, 20),
        (3, 30),
    ]
    for seed, node_count in cases:
        graph = _build_random_graph(seed, node_count=node_count)
        nx_flow = nx.min_cost_flow(graph)
        flow = min_cost_flow(graph)
        _assert_flow_feasible(graph, flow)
        assert min_cost_flow_cost(graph, flow) == nx.cost_of_flow(graph, nx_flow)


def test_regression_seeds_match_networkx():
    for seed in REGRESSION_SEEDS:
        graph = _build_random_graph(seed)
        nx_flow = nx.min_cost_flow(graph)
        flow = min_cost_flow(graph)
        _assert_flow_feasible(graph, flow)
        assert min_cost_flow_cost(graph, flow) == nx.cost_of_flow(graph, nx_flow)


@given(
    seed=st.integers(min_value=0, max_value=50_000),
    node_count=st.integers(min_value=5, max_value=25),
)
@settings(max_examples=20)
def test_property_feasible_flow(seed: int, node_count: int) -> None:
    graph = _build_random_graph(seed, node_count=node_count)
    flow = min_cost_flow(graph)
    _assert_flow_feasible(graph, flow)
