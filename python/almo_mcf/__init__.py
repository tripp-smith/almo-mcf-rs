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
"""

from ._version import __version__
from .nx import min_cost_flow, min_cost_flow_cost
from .typing import FlowDict

__all__ = ["FlowDict", "min_cost_flow", "min_cost_flow_cost", "__version__"]
