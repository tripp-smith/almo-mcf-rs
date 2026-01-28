import networkx as nx
import pytest

from almo_mcf import min_cost_flow


def test_infeasible_raises():
    G = nx.DiGraph()
    G.add_node(0, demand=-5)
    G.add_node(1, demand=5)
    G.add_edge(0, 1, capacity=2, weight=1)

    with pytest.raises(RuntimeError):
        min_cost_flow(G)
