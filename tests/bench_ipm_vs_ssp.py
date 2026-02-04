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


def summarize_metric(stats: Iterable[dict], key: str) -> float:
    values = [stat.get(key, 0.0) or 0.0 for stat in stats]
    if not values:
        return 0.0
    return statistics.median(values)


def run_case(
    node_count: int,
    edge_count: int,
    max_capacity: int,
    max_cost: int,
    seed: int,
    runs: int,
    threads: list[int],
    deterministic: bool,
    plot: bool,
) -> None:
    graph = build_benchmark_graph(node_count, edge_count, max_capacity, max_cost, seed)

    def run_ipm(thread_count: int):
        flow, stats = min_cost_flow(
            graph,
            use_ipm=True,
            strategy="periodic_rebuild",
            rebuild_every=8,
            max_iters=200,
            tolerance=1e-6,
            seed=seed,
            threads=thread_count,
            deterministic=deterministic,
            return_stats=True,
        )
        return flow, stats

    def run_classic() -> None:
        flow = min_cost_flow(graph, use_ipm=False)
        _ = min_cost_flow_cost(graph, flow)

    thread_results: dict[int, dict] = {}
    base_median = None
    base_cost = None
    for thread_count in threads:
        effective_threads = 1 if deterministic else thread_count
        timings = []
        stats_list = []
        costs = []
        for _ in range(runs):
            start = time.perf_counter()
            flow, stats = run_ipm(effective_threads)
            timings.append(time.perf_counter() - start)
            stats_list.append(stats or {})
            costs.append(min_cost_flow_cost(graph, flow))
        median_time = statistics.median(timings)
        if base_median is None and effective_threads == 1:
            base_median = median_time
            base_cost = statistics.median(costs)
        thread_results[thread_count] = {
            "timings": timings,
            "stats": stats_list,
            "costs": costs,
            "median_time": median_time,
        }

    ipm_timings = (
        thread_results[threads[0]]["timings"] if threads else []
    )
    classic_timings = time_call(run_classic, runs)

    print(
        f"nodes={node_count} edges={edge_count} "
        f"U={max_capacity} C={max_cost} seed={seed}"
    )
    if threads:
        for thread_count in threads:
            result = thread_results[thread_count]
            median_time = result["median_time"]
            speedup = (base_median / median_time) if base_median else 1.0
            cycle_ms = summarize_metric(result["stats"], "cycle_scoring_ms")
            barrier_ms = summarize_metric(result["stats"], "barrier_compute_ms")
            spanner_ms = summarize_metric(result["stats"], "spanner_update_ms")
            print(
                "  "
                + summarize(f"IPM (threads={thread_count})", result["timings"])
                + f", speedup {speedup:.2f}x"
            )
            print(
                f"    cycle_scoring_ms={cycle_ms:.2f} "
                f"barrier_compute_ms={barrier_ms:.2f} "
                f"spanner_update_ms={spanner_ms:.2f}"
            )
            if base_cost is not None:
                cost_diff = abs(statistics.median(result["costs"]) - base_cost)
                if cost_diff > 1e-6:
                    print(
                        f"    warning: cost diff {cost_diff:.3e} vs baseline threads=1"
                    )
    else:
        print("  " + summarize("IPM", ipm_timings))
    print("  " + summarize("SSP", classic_timings))

    if plot and threads:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("  warning: matplotlib not available, skipping plot")
        else:
            thread_axis = threads
            medians = [thread_results[t]["median_time"] for t in thread_axis]
            plt.figure()
            plt.plot(thread_axis, medians, marker="o")
            plt.xlabel("Threads")
            plt.ylabel("Median runtime (s)")
            plt.title(f"Scaling (n={node_count}, m={edge_count})")
            filename = (
                f"bench_scaling_n{node_count}_m{edge_count}_"
                f"cap{max_capacity}_cost{max_cost}.png"
            )
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  plot saved to {filename}")


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
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
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
                        args.threads,
                        args.deterministic,
                        not args.no_plot,
                    )


if __name__ == "__main__":
    main()
