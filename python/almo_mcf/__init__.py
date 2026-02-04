"""Python interface for almo-mcf.

Example:
    >>> import networkx as nx
    >>> from almo_mcf import min_cost_flow, min_cost_flow_cost
    >>> graph = nx.DiGraph()
    >>> graph.add_node("s", demand=-1)
    >>> graph.add_node("t", demand=1)
    >>> graph.add_edge("s", "t", capacity=5, weight=2)
    >>> flow = min_cost_flow(graph)
    >>> min_cost_flow_cost(graph, flow)
    2

Numerical tuning:
    The `min_cost_flow` entrypoint accepts parameters like `tolerance`,
    `numerical_clamp_log`, and `residual_min` to guard IPM barrier computations
    on extreme instances.
"""

from ._version import __version__
from .extensions import (
    bipartite_min_cost_matching,
    find_negative_cycle,
    gomory_hu_tree,
    isotonic_regression_dag,
    max_flow_via_min_cost_circulation,
    min_cost_flow_convex,
    sparsest_cut,
    vertex_connectivity,
)
from .nx import min_cost_flow, min_cost_flow_cost, min_cost_flow_scaled
from .typing import FlowDict

__all__ = [
    "FlowDict",
    "bipartite_min_cost_matching",
    "find_negative_cycle",
    "gomory_hu_tree",
    "isotonic_regression_dag",
    "max_flow_via_min_cost_circulation",
    "min_cost_flow",
    "min_cost_flow_convex",
    "min_cost_flow_cost",
    "min_cost_flow_scaled",
    "sparsest_cut",
    "vertex_connectivity",
    "__version__",
]
