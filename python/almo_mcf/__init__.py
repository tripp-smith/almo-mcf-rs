"""Python interface for almo-mcf."""
from importlib.metadata import PackageNotFoundError, version

from .nx import min_cost_flow, min_cost_flow_cost

try:
    __version__ = version("almo-mcf")
except PackageNotFoundError:  # pragma: no cover - fallback for editable/dev installs.
    __version__ = "0.0.0"

__all__ = ["min_cost_flow", "min_cost_flow_cost", "__version__"]
