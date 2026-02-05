"""Quickstart example for almo-mcf with NetworkX."""
import networkx as nx

from almo_mcf import min_cost_flow, min_cost_flow_cost


def main() -> None:
    graph = nx.DiGraph()
    graph.add_node("s", demand=-2)
    graph.add_node("t", demand=2)
    graph.add_edge("s", "t", capacity=5, weight=3)

    flow = min_cost_flow(graph)
    print("flow:", flow)
    print("cost:", min_cost_flow_cost(graph, flow))


if __name__ == "__main__":
    main()
