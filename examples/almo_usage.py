"""Example usage for almost-linear min-cost flow."""
from __future__ import annotations

import networkx as nx

from almo_mcf import min_cost_flow, min_cost_flow_cost


def build_grid_graph(size: int = 8) -> nx.DiGraph:
    graph = nx.DiGraph()
    for i in range(size):
        for j in range(size):
            node = (i, j)
            graph.add_node(node, demand=0)
            if i + 1 < size:
                graph.add_edge(node, (i + 1, j), capacity=5, weight=1)
            if j + 1 < size:
                graph.add_edge(node, (i, j + 1), capacity=5, weight=1)

    graph.nodes[(0, 0)]["demand"] = -10
    graph.nodes[(size - 1, size - 1)]["demand"] = 10
    return graph


def main() -> None:
    graph = build_grid_graph()
    flow, stats = min_cost_flow(
        graph,
        use_ipm=True,
        deterministic=True,
        return_stats=True,
    )
    print("solver_mode:", stats["solver_mode"])
    print("cost:", min_cost_flow_cost(graph, flow))


if __name__ == "__main__":
    main()
