import networkx as nx

from almo_mcf import min_cost_flow, min_cost_flow_cost


def test_single_path():
    G = nx.DiGraph()
    G.add_node("s", demand=-3)
    G.add_node("t", demand=3)
    G.add_edge("s", "t", capacity=5, weight=2)

    flow = min_cost_flow(G)
    assert flow["s"]["t"] == 3
    assert min_cost_flow_cost(G, flow) == 6


def test_parallel_path_preference():
    G = nx.DiGraph()
    G.add_node(0, demand=-4)
    G.add_node(1, demand=0)
    G.add_node(2, demand=4)
    G.add_edge(0, 2, capacity=2, weight=1)
    G.add_edge(0, 1, capacity=4, weight=2)
    G.add_edge(1, 2, capacity=4, weight=2)

    flow = min_cost_flow(G)
    assert flow[0][2] == 2
    assert flow[0][1] == 2
    assert flow[1][2] == 2
    assert min_cost_flow_cost(G, flow) == 10


def test_lower_bounds():
    G = nx.DiGraph()
    G.add_node(0, demand=-3)
    G.add_node(1, demand=0)
    G.add_node(2, demand=3)
    G.add_edge(0, 1, capacity=5, lower_capacity=2, weight=1)
    G.add_edge(1, 2, capacity=4, lower_capacity=1, weight=2)

    flow = min_cost_flow(G)
    assert flow[0][1] == 3
    assert flow[1][2] == 3
    assert min_cost_flow_cost(G, flow) == 9


def test_negative_costs():
    G = nx.DiGraph()
    G.add_node("a", demand=-2)
    G.add_node("b", demand=2)
    G.add_edge("a", "b", capacity=3, weight=-1)

    flow = min_cost_flow(G)
    assert flow["a"]["b"] == 2
    assert min_cost_flow_cost(G, flow) == -2
