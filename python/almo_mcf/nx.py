"""NetworkX adapter for almo-mcf.

This module provides the public functions expected by the README specification.
The actual solver is expected to be provided by the Rust extension module when
available.
"""
from __future__ import annotations

from typing import Dict


def _load_core():
    try:
        from . import _core  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised when extension is absent
        raise RuntimeError(
            "almo-mcf core extension is not built. "
            "Build the Rust extension via maturin to use the solver."
        ) from exc
    return _core


def min_cost_flow(G) -> Dict:
    """Return a min-cost flow dict in NetworkX format.

    Args:
        G: NetworkX DiGraph with demand and capacity attributes.
    """
    core = _load_core()
    return core.min_cost_flow_nx(G)


def min_cost_flow_cost(G, flow_dict: Dict) -> int:
    """Compute the total cost for a flow dict in NetworkX format."""
    core = _load_core()
    return core.min_cost_flow_cost_nx(G, flow_dict)
