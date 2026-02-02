import networkx as nx

from almo_mcf.nx import min_cost_flow, min_cost_flow_cost


def _build_graph():
    g = nx.DiGraph()
    g.add_node("s", demand=-3)
    g.add_node("t", demand=3)
    g.add_edge("s", "t", capacity=4, weight=2)
    g.add_edge("s", "t2", capacity=2, weight=3)
    g.add_node("t2", demand=0)
    g.add_edge("t2", "t", capacity=2, weight=1)
    return g


def _build_dense_graph():
    g = nx.DiGraph()
    g.add_nodes_from(range(5))
    for node in range(5):
        g.nodes[node]["demand"] = 0
    g.nodes[0]["demand"] = -4
    g.nodes[4]["demand"] = 4
    edges = [
        (0, 1, 6, 2),
        (1, 2, 5, -1),
        (2, 4, 6, 3),
        (0, 3, 4, 1),
        (3, 4, 4, 2),
        (1, 3, 3, 0),
        (2, 3, 2, -2),
    ]
    for u, v, capacity, weight in edges:
        g.add_edge(u, v, capacity=capacity, weight=weight)
    return g


def test_deterministic_option_stabilizes_cycle_selection():
    g = _build_graph()
    flow_a = min_cost_flow(g, seed=7, deterministic=True)
    flow_b = min_cost_flow(g, seed=999, deterministic=True)
    assert flow_a == flow_b


def test_default_deterministic_is_stable_across_seeds():
    g = _build_graph()
    flow_a = min_cost_flow(g, seed=1)
    flow_b = min_cost_flow(g, seed=2)
    assert flow_a == flow_b


def test_deterministic_ipm_is_reproducible_across_seeds():
    g = _build_dense_graph()
    flow_a, stats_a = min_cost_flow(
        g,
        seed=10,
        deterministic=True,
        use_ipm=True,
        max_iters=30,
        tolerance=1e-6,
        return_stats=True,
    )
    flow_b, stats_b = min_cost_flow(
        g,
        seed=999,
        deterministic=True,
        use_ipm=True,
        max_iters=30,
        tolerance=1e-6,
        return_stats=True,
    )
    assert flow_a == flow_b
    assert stats_a is not None
    assert stats_b is not None


def test_deterministic_matches_networkx_cost_for_exact_flow():
    g = _build_dense_graph()
    flow = min_cost_flow(
        g,
        deterministic=True,
        use_ipm=True,
        max_iters=50,
        tolerance=1e-6,
    )
    assert min_cost_flow_cost(g, flow) == nx.cost_of_flow(g, nx.min_cost_flow(g))
