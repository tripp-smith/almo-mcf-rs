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
    graph = build_benchmark_graph(12, 60)

    def run() -> dict:
        return min_cost_flow(
            graph,
            strategy="periodic_rebuild",
            rebuild_every=2,
            max_iters=50,
            tolerance=1e-6,
            seed=3,
            threads=2,
            alpha=0.0005,
        )

    flow = benchmark.pedantic(run, rounds=1, iterations=1)
    nx_flow = nx.min_cost_flow(graph)
    assert nx.cost_of_flow(graph, nx_flow) == nx.cost_of_flow(graph, flow)


@pytest.mark.skipif(
    os.getenv("RUN_BENCHMARKS") != "1", reason="benchmarks disabled by default"
)
def test_networkx_benchmark(benchmark) -> None:
    graph = build_benchmark_graph(12, 60)

    def run() -> dict:
        return nx.min_cost_flow(graph)

    benchmark.pedantic(run, rounds=1, iterations=1)


@pytest.mark.skipif(
    os.getenv("RUN_BENCHMARKS") != "1", reason="benchmarks disabled by default"
)
@pytest.mark.parametrize(
    ("node_count", "edge_count"),
    [(12, 60), (20, 120)],
)
def test_deterministic_vs_randomized_benchmark(
    benchmark, node_count: int, edge_count: int
) -> None:
    graph = build_benchmark_graph(node_count, edge_count)

    def run_deterministic() -> dict:
        return min_cost_flow(
            graph,
            use_ipm=True,
            deterministic=True,
            strategy="periodic_rebuild",
            rebuild_every=2,
            max_iters=50,
            tolerance=1e-6,
            seed=11,
            threads=2,
            alpha=0.0005,
        )

    def run_randomized() -> dict:
        return min_cost_flow(
            graph,
            use_ipm=True,
            deterministic=False,
            strategy="periodic_rebuild",
            rebuild_every=2,
            max_iters=50,
            tolerance=1e-6,
            seed=11,
            threads=2,
            alpha=0.0005,
        )

    flow = benchmark.pedantic(run_deterministic, rounds=1, iterations=1)
    randomized_flow = benchmark.pedantic(run_randomized, rounds=1, iterations=1)
    nx_flow = nx.min_cost_flow(graph)
    assert nx.cost_of_flow(graph, flow) == nx.cost_of_flow(graph, nx_flow)
    assert nx.cost_of_flow(graph, randomized_flow) == nx.cost_of_flow(graph, nx_flow)
