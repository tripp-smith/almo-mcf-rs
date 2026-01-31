#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import networkx as nx

from almo_mcf import min_cost_flow, min_cost_flow_cost


def build_benchmark_graph(node_count: int, edge_count: int, seed: int) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(node_count))
    for node in range(node_count):
        graph.nodes[node]["demand"] = 0
    graph.nodes[0]["demand"] = -15
    graph.nodes[node_count - 1]["demand"] = 15

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
        head = (idx * 7 + 3 + seed) % node_count
        if tail == head:
            head = (head + 1) % node_count
        graph.add_edge(
            tail,
            head,
            capacity=15,
            lower_capacity=0,
            weight=((idx * 13 + seed) % 11) - 5,
        )
    return graph


def time_call(fn: Callable[[], object], runs: int) -> list[float]:
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark almo-mcf vs NetworkX.")
    parser.add_argument("--nodes", type=int, default=40)
    parser.add_argument("--edges", type=int, default=200)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--strategy", type=str, default="periodic_rebuild")
    parser.add_argument("--rebuild-every", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--skip-networkx", action="store_true")
    args = parser.parse_args()

    graph = build_benchmark_graph(args.nodes, args.edges, args.seed)

    def run_almo() -> None:
        flow, _stats = min_cost_flow(
            graph,
            strategy=args.strategy,
            rebuild_every=args.rebuild_every,
            max_iters=args.max_iters,
            tolerance=args.tolerance,
            seed=args.seed,
            threads=args.threads,
            alpha=args.alpha,
            return_stats=True,
        )
        _ = min_cost_flow_cost(graph, flow)

    almo_timings = time_call(run_almo, args.runs)
    print(
        "almo-mcf timings (s):",
        ", ".join(f"{t:.4f}" for t in almo_timings),
        f"(median {statistics.median(almo_timings):.4f}s)",
    )

    if not args.skip_networkx:
        def run_networkx() -> None:
            flow = nx.min_cost_flow(graph)
            _ = nx.cost_of_flow(graph, flow)

        nx_timings = time_call(run_networkx, args.runs)
        print(
            "networkx timings (s):",
            ", ".join(f"{t:.4f}" for t in nx_timings),
            f"(median {statistics.median(nx_timings):.4f}s)",
        )


if __name__ == "__main__":
    main()
