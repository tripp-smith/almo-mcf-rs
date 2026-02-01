import os

import networkx as nx
import pytest

from almo_mcf import min_cost_flow
from almo_mcf import nx as almo_nx


def _build_large_graph(
    node_count: int = 400, edge_count: int = 100_000
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(node_count))
    for node in range(node_count):
        graph.nodes[node]["demand"] = 0
    graph.nodes[0]["demand"] = -1
    graph.nodes[node_count - 1]["demand"] = 1

    for node in range(node_count - 1):
        graph.add_edge(node, node + 1, capacity=2, lower_capacity=0, weight=1)

    remaining = edge_count - (node_count - 1)
    for idx in range(remaining):
        tail = idx % node_count
        head = (idx * 37 + 11) % node_count
        if tail == head:
            head = (head + 1) % node_count
        graph.add_edge(tail, head, capacity=3, lower_capacity=0, weight=(idx % 5) - 2)

    return graph


@pytest.mark.skipif(
    os.getenv("RUN_SCALABILITY") != "1", reason="scalability test disabled by default"
)
def test_large_graph_ipm_smoke() -> None:
    graph = _build_large_graph()
    tails, heads, lower, upper, cost, demand, _index, _edges = almo_nx._graph_to_arrays(
        graph
    )
    lower, upper, cost, demand, capacity_scale = almo_nx._scale_problem(
        lower, upper, cost, demand
    )
    assert len(tails) == len(heads) == len(lower) == len(upper) == len(cost)
    assert len(tails) >= 100_000
    assert capacity_scale >= 1

    if os.getenv("RUN_SCALABILITY_SOLVER") == "1":
        flow, stats = min_cost_flow(
            graph,
            use_ipm=True,
            strategy="periodic_rebuild",
            rebuild_every=20,
            max_iters=1,
            tolerance=1e9,
            seed=17,
            threads=1,
            alpha=0.0005,
            return_stats=True,
        )
        assert stats is not None
        assert stats["termination"] in {
            "converged",
            "iteration_limit",
            "time_limit",
            "no_improving_cycle",
        }
        assert flow[0][1] >= 0
