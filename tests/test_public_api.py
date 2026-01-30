from almo_mcf import __version__
from almo_mcf import typing as almo_typing
from almo_mcf._version import __version__ as module_version


def test_version_alias_matches_module():
    assert __version__ == module_version


def test_typing_aliases_exported():
    assert almo_typing.FlowDict is not None
    assert almo_typing.Node is not None
