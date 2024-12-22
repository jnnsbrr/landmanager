import os
import pytest


@pytest.fixture
def test_path():
    """Fixture for the test path."""
    return os.path.dirname(os.path.abspath(__file__))


def pytest_configure(config):
    import sys

    sys._called_from_test = True


def pytest_unconfigure(config):
    import sys  # This was missing from the manual

    del sys._called_from_test
