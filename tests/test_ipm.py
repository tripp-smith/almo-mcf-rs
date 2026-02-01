import networkx as nx
import numpy as np

from almo_mcf import min_cost_flow, min_cost_flow_cost
from almo_mcf import _core


def _build_large_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    node_count = 10
    for node in range(node_count):
        graph.add_node(node, demand=0)

    graph.nodes[0]["demand"] = -5
    graph.nodes[5]["demand"] = 5

    for node in range(node_count):
        next_node = (node + 1) % node_count
        graph.add_edge(node, next_node, capacity=6, weight=0)

    extra_edges = [
        (0, 2, 3),
        (2, 5, 4),
        (5, 7, 3),
        (7, 9, 4),
        (1, 6, 2),
    ]
    for u, v, capacity in extra_edges:
        graph.add_edge(u, v, capacity=capacity, weight=0)

    return graph


def _assert_flow_integral(graph: nx.DiGraph, flow) -> None:
    for u, v, data in graph.edges(data=True):
        value = flow[u][v]
        assert float(value).is_integer()
        assert 0 <= value <= data["capacity"]

    for node in graph.nodes():
        inflow = sum(flow[u][node] for u in graph.predecessors(node))
        outflow = sum(flow[node][v] for v in graph.successors(node))
        assert inflow - outflow == graph.nodes[node].get("demand", 0)


def test_ipm_rounding_produces_integral_flow():
    graph = _build_large_graph()
    flow, stats = min_cost_flow(
        graph,
        use_ipm=True,
        strategy="full_dynamic",
        max_iters=50,
        tolerance=1e6,
        seed=7,
        threads=2,
        alpha=0.0005,
        return_stats=True,
    )
    assert stats is not None
    assert set(stats.keys()) >= {"iterations", "final_gap", "termination"}
    _assert_flow_integral(graph, flow)
    assert min_cost_flow_cost(graph, flow) == nx.cost_of_flow(graph, nx.min_cost_flow(graph))


def test_run_ipm_full_dynamic_emits_stats():
    graph = _build_large_graph()
    nodes = list(graph.nodes())
    index = {node: idx for idx, node in enumerate(nodes)}
    edges = list(graph.edges(data=True))
    tails = np.asarray([index[u] for u, _v, _ in edges], dtype=np.int64)
    heads = np.asarray([index[v] for _u, v, _ in edges], dtype=np.int64)
    lower = np.zeros(len(edges), dtype=np.int64)
    upper = np.asarray([data["capacity"] for _u, _v, data in edges], dtype=np.int64)
    cost = np.asarray([data["weight"] for _u, _v, data in edges], dtype=np.int64)
    demand = np.asarray([graph.nodes[node]["demand"] for node in nodes], dtype=np.int64)

    flow, stats = _core.run_ipm_edges(
        len(nodes),
        tails,
        heads,
        lower,
        upper,
        cost,
        demand,
        strategy="full_dynamic",
        max_iters=1,
        tolerance=0.0,
        seed=11,
        threads=2,
        alpha=0.0005,
    )
    assert flow.shape[0] == len(edges)
    assert stats["termination"] in {
        "converged",
        "iteration_limit",
        "time_limit",
        "no_improving_cycle",
    }


def test_run_ipm_periodic_rebuild_emits_stats():
    graph = _build_large_graph()
    nodes = list(graph.nodes())
    index = {node: idx for idx, node in enumerate(nodes)}
    edges = list(graph.edges(data=True))
    tails = np.asarray([index[u] for u, _v, _ in edges], dtype=np.int64)
    heads = np.asarray([index[v] for _u, v, _ in edges], dtype=np.int64)
    lower = np.zeros(len(edges), dtype=np.int64)
    upper = np.asarray([data["capacity"] for _u, _v, data in edges], dtype=np.int64)
    cost = np.asarray([data["weight"] for _u, _v, data in edges], dtype=np.int64)
    demand = np.asarray([graph.nodes[node]["demand"] for node in nodes], dtype=np.int64)

    flow, stats = _core.run_ipm_edges(
        len(nodes),
        tails,
        heads,
        lower,
        upper,
        cost,
        demand,
        strategy="periodic_rebuild",
        rebuild_every=3,
        max_iters=1,
        tolerance=0.0,
        seed=19,
        threads=2,
        alpha=0.0005,
    )
    assert flow.shape[0] == len(edges)
    assert stats["termination"] in {
        "converged",
        "iteration_limit",
        "time_limit",
        "no_improving_cycle",
    }


def test_use_ipm_flag_toggles_solver_path():
    graph = _build_large_graph()
    flow, stats = min_cost_flow(
        graph,
        use_ipm=False,
        return_stats=True,
    )
    assert stats is None
    _assert_flow_integral(graph, flow)

    flow, stats = min_cost_flow(
        graph,
        use_ipm=True,
        max_iters=5,
        return_stats=True,
    )
    assert stats is not None
    _assert_flow_integral(graph, flow)
