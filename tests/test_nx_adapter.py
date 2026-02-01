import math

import networkx as nx
import pytest

from almo_mcf import nx as almo_nx


def test_graph_requires_directed():
    graph = nx.Graph()
    graph.add_node("a")
    with pytest.raises(ValueError, match="directed"):
        almo_nx._graph_to_arrays(graph)


def test_graph_accepts_multidigraph():
    graph = nx.MultiDiGraph()
    graph.add_edge("a", "b", capacity=1, weight=2)
    graph.add_edge("a", "b", capacity=3, weight=4)
    tails, heads, lower, upper, cost, demand, index, edges = almo_nx._graph_to_arrays(
        graph
    )
    assert len(edges) == 2
    assert tails == [index["a"], index["a"]]
    assert heads == [index["b"], index["b"]]
    assert cost == [2, 4]


def test_missing_capacity_is_error():
    graph = nx.DiGraph()
    graph.add_edge("a", "b")
    with pytest.raises(ValueError, match="capacity"):
        almo_nx._graph_to_arrays(graph)


def test_infinite_capacity_is_error():
    graph = nx.DiGraph()
    graph.add_edge("a", "b", capacity=math.inf)
    with pytest.raises(ValueError, match="capacity"):
        almo_nx._graph_to_arrays(graph)


def test_non_finite_demand_is_error():
    graph = nx.DiGraph()
    graph.add_node("a", demand=math.inf)
    graph.add_node("b", demand=0)
    graph.add_edge("a", "b", capacity=3)
    with pytest.raises(ValueError, match="demand"):
        almo_nx._graph_to_arrays(graph)


def test_lower_capacity_parsed():
    graph = nx.DiGraph()
    graph.add_node("a", demand=-1)
    graph.add_node("b", demand=1)
    graph.add_edge("a", "b", capacity=5, lower_capacity=2, weight=3)
    tails, heads, lower, upper, cost, demand, index, edges = almo_nx._graph_to_arrays(
        graph
    )
    assert tails == [index["a"]]
    assert heads == [index["b"]]
    assert lower == [2]
    assert upper == [5]
    assert cost == [3]
    assert demand[index["a"]] == -1
    assert demand[index["b"]] == 1
    assert edges[0][0:2] == ("a", "b")
