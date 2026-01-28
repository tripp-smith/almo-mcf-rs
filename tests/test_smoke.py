import almo_mcf


def test_module_version_present():
    assert hasattr(almo_mcf, "__version__")
