#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import statistics
import time
from typing import Iterable

import networkx as nx

from almo_mcf import min_cost_flow, min_cost_flow_cost


def build_benchmark_graph(
    node_count: int,
    edge_count: int,
    max_capacity: int,
    max_cost: int,
    seed: int,
) -> nx.DiGraph:
    rng = random.Random(seed)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(node_count))

    for idx in range(node_count):
        next_idx = (idx + 1) % node_count
        graph.add_edge(
            idx,
            next_idx,
            capacity=rng.randint(1, max_capacity),
            weight=rng.randint(-max_cost, max_cost),
        )

    while graph.number_of_edges() < edge_count:
        tail = rng.randrange(node_count)
        head = rng.randrange(node_count)
        if tail == head:
            continue
        if graph.has_edge(tail, head):
            continue
        graph.add_edge(
            tail,
            head,
            capacity=rng.randint(1, max_capacity),
            weight=rng.randint(-max_cost, max_cost),
        )

    demands = [0] * node_count
    for u, v, data in graph.edges(data=True):
        flow_value = rng.randint(0, min(max_capacity, data["capacity"]))
        demands[u] -= flow_value
        demands[v] += flow_value

    for node, demand in enumerate(demands):
        graph.nodes[node]["demand"] = demand

    return graph


def time_call(fn, runs: int) -> list[float]:
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return timings


def summarize(name: str, timings: Iterable[float]) -> str:
    timings = list(timings)
    return (
        f"{name}: median {statistics.median(timings):.4f}s "
        f"(min {min(timings):.4f}s, max {max(timings):.4f}s)"
    )


def run_case(
    node_count: int,
    edge_count: int,
    max_capacity: int,
    max_cost: int,
    seed: int,
    runs: int,
) -> None:
    graph = build_benchmark_graph(node_count, edge_count, max_capacity, max_cost, seed)

    def run_ipm() -> None:
        flow, _stats = min_cost_flow(
            graph,
            use_ipm=True,
            strategy="periodic_rebuild",
            rebuild_every=8,
            max_iters=200,
            tolerance=1e-6,
            seed=seed,
            threads=2,
            return_stats=True,
        )
        _ = min_cost_flow_cost(graph, flow)

    def run_classic() -> None:
        flow = min_cost_flow(graph, use_ipm=False)
        _ = min_cost_flow_cost(graph, flow)

    ipm_timings = time_call(run_ipm, runs)
    classic_timings = time_call(run_classic, runs)

    print(
        f"nodes={node_count} edges={edge_count} "
        f"U={max_capacity} C={max_cost} seed={seed}"
    )
    print("  " + summarize("IPM", ipm_timings))
    print("  " + summarize("SSP", classic_timings))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark IPM vs successive shortest path across sizes."
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--nodes", type=int, nargs="+", default=[30, 60])
    parser.add_argument("--edges", type=int, nargs="+", default=[150, 300])
    parser.add_argument("--capacity", type=int, nargs="+", default=[10, 50])
    parser.add_argument("--cost", type=int, nargs="+", default=[5, 20])
    args = parser.parse_args()

    for node_count in args.nodes:
        for edge_count in args.edges:
            for max_capacity in args.capacity:
                for max_cost in args.cost:
                    run_case(
                        node_count,
                        edge_count,
                        max_capacity,
                        max_cost,
                        args.seed,
                        args.runs,
                    )


if __name__ == "__main__":
    main()
