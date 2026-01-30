"""Version helpers for almo-mcf."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("almo-mcf")
except PackageNotFoundError:  # pragma: no cover - fallback for editable/dev installs.
    __version__ = "0.0.0"

__all__ = ["__version__"]
