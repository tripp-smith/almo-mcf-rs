import os
import random

import networkx as nx
import pytest
from almo_mcf import min_cost_flow, min_cost_flow_cost


def _build_random_graph(
    seed: int,
    node_count: int,
    edge_count: int,
    max_capacity: int,
    max_cost: int,
) -> nx.DiGraph:
    rng = random.Random(seed)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(node_count))

    for idx in range(node_count):
        next_idx = (idx + 1) % node_count
        graph.add_edge(
            idx,
            next_idx,
            capacity=rng.randint(1, max_capacity),
            weight=rng.randint(-max_cost, max_cost),
        )

    while graph.number_of_edges() < edge_count:
        tail = rng.randrange(node_count)
        head = rng.randrange(node_count)
        if tail == head or graph.has_edge(tail, head):
            continue
        graph.add_edge(
            tail,
            head,
            capacity=rng.randint(1, max_capacity),
            weight=rng.randint(-max_cost, max_cost),
        )

    demands = [0] * node_count
    for u, v, data in graph.edges(data=True):
        flow_value = rng.randint(0, data["capacity"])
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


@pytest.mark.skipif(
    os.getenv("RUN_RANDOM_PARITY") != "1",
    reason="random parity suite disabled by default",
)
def test_ipm_matches_networkx_on_random_graphs():
    seed, nodes, edges, capacity, cost = (11, 6, 10, 6, 6)
    graph = _build_random_graph(seed, nodes, edges, capacity, cost)
    nx_flow = nx.min_cost_flow(graph)
    flow = min_cost_flow(graph, use_ipm=False)
    _assert_flow_feasible(graph, flow)
    assert min_cost_flow_cost(graph, flow) == nx.cost_of_flow(graph, nx_flow)


@pytest.mark.skipif(
    os.getenv("RUN_RANDOM_PARITY") != "1",
    reason="random parity suite disabled by default",
)
def test_large_instance_regression():
    seed = 101
    graph = _build_random_graph(seed, 8, 12, 10, 8)
    nx_flow = nx.min_cost_flow(graph)
    flow = min_cost_flow(graph, use_ipm=False)
    _assert_flow_feasible(graph, flow)
    assert min_cost_flow_cost(graph, flow) == nx.cost_of_flow(graph, nx_flow)
