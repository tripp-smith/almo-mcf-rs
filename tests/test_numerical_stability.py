import math

import networkx as nx

from almo_mcf import min_cost_flow, min_cost_flow_cost


def _assert_flow_within_bounds(graph: nx.DiGraph, flow) -> None:
    for u, v, data in graph.edges(data=True):
        value = flow[u][v]
        assert math.isfinite(value)
        assert data.get("lower_capacity", 0) <= value <= data["capacity"]


def test_near_bound_flow_is_stable():
    graph = nx.DiGraph()
    graph.add_node(0, demand=-5)
    graph.add_node(1, demand=0)
    graph.add_node(2, demand=5)
    graph.add_edge(0, 1, capacity=5, lower_capacity=4, weight=1)
    graph.add_edge(1, 2, capacity=5, lower_capacity=4, weight=1)
    graph.add_edge(0, 2, capacity=1, lower_capacity=0, weight=10)

    flow = min_cost_flow(graph)
    _assert_flow_within_bounds(graph, flow)
    assert min_cost_flow_cost(graph, flow) == 10


def test_large_near_bound_capacities():
    graph = nx.DiGraph()
    graph.add_node("s", demand=-999_999)
    graph.add_node("t", demand=999_999)
    graph.add_edge("s", "t", capacity=1_000_000, lower_capacity=999_999, weight=3)

    flow = min_cost_flow(graph)
    _assert_flow_within_bounds(graph, flow)
    assert flow["s"]["t"] == 999_999
