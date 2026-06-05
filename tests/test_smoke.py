import vectormark


def test_version_exposed():
    assert isinstance(vectormark.__version__, str)
    assert vectormark.__version__.count(".") == 2
