"""Type aliases for the almo-mcf public API."""
from __future__ import annotations

from typing import Dict, Hashable, Mapping

Node = Hashable
Cost = int
Demand = int
Capacity = int
FlowValue = int

Edge = tuple[Node, Node]
FlowDict = Dict[Node, Dict[Node, FlowValue]]
NodeDemandMap = Mapping[Node, Demand]
EdgeCostMap = Mapping[Edge, Cost]
EdgeCapacityMap = Mapping[Edge, Capacity]

__all__ = [
    "Capacity",
    "Cost",
    "Demand",
    "Edge",
    "EdgeCapacityMap",
    "EdgeCostMap",
    "FlowDict",
    "FlowValue",
    "Node",
    "NodeDemandMap",
]
