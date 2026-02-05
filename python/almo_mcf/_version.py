"""Version helpers for almo-mcf."""
from importlib.metadata import PackageNotFoundError, version

__version__ = "0.2.0"
try:
    __version__ = version("almo-mcf")
except PackageNotFoundError:  # pragma: no cover - fallback for editable/dev installs.
    pass

__all__ = ["__version__"]
