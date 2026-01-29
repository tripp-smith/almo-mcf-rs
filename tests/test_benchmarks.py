import os

import networkx as nx
import pytest

from almo_mcf import min_cost_flow


def build_benchmark_graph(node_count: int, edge_count: int) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(node_count))
    for node in range(node_count):
        graph.nodes[node]["demand"] = 0
    graph.nodes[0]["demand"] = -10
    graph.nodes[node_count - 1]["demand"] = 10

    for node in range(node_count - 1):
        graph.add_edge(
            node,
            node + 1,
            capacity=20,
            lower_capacity=0,
            weight=1,
        )

    for idx in range(edge_count):
        tail = idx % node_count
        head = (idx * 7 + 3) % node_count
        if tail == head:
            head = (head + 1) % node_count
        graph.add_edge(
            tail,
            head,
            capacity=15,
            lower_capacity=0,
            weight=((idx * 13) % 11) - 5,
        )
    return graph


@pytest.mark.skipif(
    os.getenv("RUN_BENCHMARKS") != "1", reason="benchmarks disabled by default"
)
def test_min_cost_flow_benchmark(benchmark) -> None:
    graph = build_benchmark_graph(6, 10)

    def run() -> None:
        min_cost_flow(graph)

    benchmark.pedantic(run, rounds=1, iterations=1)
