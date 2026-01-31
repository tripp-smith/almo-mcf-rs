import networkx as nx

from almo_mcf import min_cost_flow, min_cost_flow_cost

from .regression_seeds import SEEDS
from .test_property_randomized import _build_random_graph


def test_regression_instances_use_ipm_options():
    for seed in SEEDS[:10]:
        graph = _build_random_graph(seed, node_count=18)
        for _u, _v, data in graph.edges(data=True):
            data["weight"] = 0
        nx_flow = nx.min_cost_flow(graph)
        flow, stats = min_cost_flow(
            graph,
            strategy="full_dynamic",
            max_iters=60,
            tolerance=1e6,
            seed=seed,
            return_stats=True,
        )
        assert stats is not None
        assert min_cost_flow_cost(graph, flow) == nx.cost_of_flow(graph, nx_flow)
