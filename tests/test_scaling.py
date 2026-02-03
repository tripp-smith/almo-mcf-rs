import networkx as nx

from almo_mcf import min_cost_flow_scaled


def test_min_cost_flow_scaled_handles_large_bounds():
    graph = nx.DiGraph()
    for node in range(9):
        graph.add_node(node, demand=0)
    for i in range(13):
        tail = i % 9
        head = (i + 1) % 9
        graph.add_edge(
            tail,
            head,
            capacity=(1 << 20) + (1 if i % 2 == 0 else 3),
            weight=(1 << 20) + i,
        )

    flow = min_cost_flow_scaled(graph)
    assert all(value == 0 for edges in flow.values() for value in edges.values())
