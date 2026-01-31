import networkx as nx

from almo_mcf import min_cost_flow, min_cost_flow_cost


def _graphs():
    graphs = []
    G = nx.DiGraph()
    G.add_node("s", demand=-4)
    G.add_node("a", demand=0)
    G.add_node("t", demand=4)
    G.add_edge("s", "a", capacity=4, weight=1)
    G.add_edge("a", "t", capacity=4, weight=1)
    graphs.append(G)

    G = nx.DiGraph()
    for node, demand in [("s", -3), ("m", 0), ("t", 3)]:
        G.add_node(node, demand=demand)
    G.add_edge("s", "m", capacity=2, weight=2)
    G.add_edge("s", "t", capacity=2, weight=5)
    G.add_edge("m", "t", capacity=2, weight=1)
    graphs.append(G)

    G = nx.DiGraph()
    for node, demand in [(0, -5), (1, 2), (2, 3)]:
        G.add_node(node, demand=demand)
    G.add_edge(0, 1, capacity=3, weight=2)
    G.add_edge(0, 2, capacity=5, weight=1)
    G.add_edge(1, 2, capacity=5, weight=3)
    graphs.append(G)
    return graphs


def test_matches_networkx_cost():
    for G in _graphs():
        nx_flow = nx.min_cost_flow(G)
        flow = min_cost_flow(G)
        assert min_cost_flow_cost(G, flow) == nx.cost_of_flow(G, nx_flow)


def test_matches_networkx_cost_with_ipm_options():
    G = nx.DiGraph()
    for node, demand in [(0, -6), (1, 0), (2, 0), (3, 0), (4, 6), (5, 0), (6, 0), (7, 0), (8, 0)]:
        G.add_node(node, demand=demand)
    for i in range(9):
        G.add_edge(i, (i + 1) % 9, capacity=6, weight=0)
    G.add_edge(0, 4, capacity=6, weight=0)
    G.add_edge(1, 5, capacity=6, weight=0)
    G.add_edge(2, 6, capacity=6, weight=0)
    G.add_edge(3, 7, capacity=6, weight=0)
    G.add_edge(4, 8, capacity=6, weight=0)

    nx_flow = nx.min_cost_flow(G)
    flow, stats = min_cost_flow(
        G,
        strategy="periodic_rebuild",
        rebuild_every=2,
        max_iters=40,
        tolerance=1e6,
        seed=3,
        return_stats=True,
    )
    assert stats is not None
    assert min_cost_flow_cost(G, flow) == nx.cost_of_flow(G, nx_flow)
