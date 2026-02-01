import math

import networkx as nx
import pytest

from almo_mcf import (
    bipartite_min_cost_matching,
    find_negative_cycle,
    gomory_hu_tree,
    isotonic_regression_dag,
    max_flow_via_min_cost_circulation,
    min_cost_flow,
    min_cost_flow_convex,
    sparsest_cut,
    vertex_connectivity,
)


def test_multidigraph_min_cost_flow_supports_parallel_edges():
    graph = nx.MultiDiGraph()
    graph.add_node("s", demand=-3)
    graph.add_node("t", demand=3)
    graph.add_edge("s", "t", capacity=2, weight=1)
    graph.add_edge("s", "t", capacity=4, weight=5)

    flow = min_cost_flow(graph)
    flows = flow["s"]["t"]
    assert sum(flows.values()) == 3
    cheap_edge_key = min(flows, key=flows.get)
    assert flows[cheap_edge_key] <= 2


def test_capacity_and_cost_scaling_handles_large_values():
    graph = nx.DiGraph()
    graph.add_node("s", demand=-1_000_000)
    graph.add_node("t", demand=1_000_000)
    graph.add_edge("s", "t", capacity=1_000_000, weight=1_000_000)

    flow = min_cost_flow(graph)
    assert flow["s"]["t"] == 1_000_000


def test_min_cost_flow_convex_respects_marginal_costs():
    graph = nx.MultiDiGraph()
    graph.add_node("s", demand=-2)
    graph.add_node("t", demand=2)
    graph.add_edge("s", "t", capacity=2, convex_cost=[0, 1, 6])
    graph.add_edge("s", "t", capacity=2, convex_cost=[0, 2, 4])

    flow = min_cost_flow_convex(graph)
    flows = flow["s"]["t"]
    assert sorted(flows.values()) == [1, 1]


def test_max_flow_via_min_cost_circulation_matches_networkx():
    graph = nx.DiGraph()
    graph.add_edge("s", "a", capacity=3)
    graph.add_edge("a", "t", capacity=2)
    graph.add_edge("s", "t", capacity=1)

    max_flow_value, _flow = max_flow_via_min_cost_circulation(
        graph, "s", "t", deterministic=True
    )
    nx_value = nx.maximum_flow_value(graph, "s", "t")
    assert max_flow_value == nx_value


def test_negative_cycle_detection():
    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=-2)
    graph.add_edge("b", "c", weight=-2)
    graph.add_edge("c", "a", weight=1)

    cycle = find_negative_cycle(graph)
    assert cycle is not None
    assert cycle[0] == cycle[-1]


def test_isotonic_regression_dag_chain():
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    values = {0: 3.0, 1: 1.0}

    solution = isotonic_regression_dag(graph, values)
    assert solution[0] == pytest.approx(2.0)
    assert solution[1] == pytest.approx(2.0)


def test_bipartite_min_cost_matching():
    graph = nx.Graph()
    left = ["l1", "l2"]
    right = ["r1", "r2"]
    graph.add_edge("l1", "r1", weight=5)
    graph.add_edge("l1", "r2", weight=1)
    graph.add_edge("l2", "r1", weight=1)
    graph.add_edge("l2", "r2", weight=5)

    matching = bipartite_min_cost_matching(graph, left, right)
    assert matching["l1"] == "r2"
    assert matching["l2"] == "r1"


def test_graph_utilities():
    graph = nx.Graph()
    graph.add_edge(0, 1, capacity=3)
    graph.add_edge(1, 2, capacity=2)
    graph.add_edge(0, 2, capacity=1)

    assert vertex_connectivity(graph) == 2
    tree = gomory_hu_tree(graph, capacity="capacity")
    assert isinstance(tree, nx.Graph)
    ratio, (part, other) = sparsest_cut(graph, capacity="capacity")
    assert ratio >= 0
    assert part or other


def test_invalid_input_validation():
    graph = nx.DiGraph()
    graph.add_node("s", demand=-1)
    graph.add_node("t", demand=0)
    graph.add_edge("s", "t", capacity=1, weight=1)
    with pytest.raises(ValueError):
        min_cost_flow(graph)

    graph = nx.DiGraph()
    graph.add_node("s", demand=-1)
    graph.add_node("t", demand=1)
    graph.add_edge("s", "t", capacity=math.inf, weight=1)
    with pytest.raises(ValueError):
        min_cost_flow(graph)
