import networkx as nx

from almo_mcf.nx import min_cost_flow


def _build_graph():
    g = nx.DiGraph()
    g.add_node("s", demand=-3)
    g.add_node("t", demand=3)
    g.add_edge("s", "t", capacity=4, weight=2)
    g.add_edge("s", "t2", capacity=2, weight=3)
    g.add_node("t2", demand=0)
    g.add_edge("t2", "t", capacity=2, weight=1)
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
