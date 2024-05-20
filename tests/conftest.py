#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Testing configuration."""

import pytest


def pytest_addoption(parser):  # noqa: D103
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):  # noqa: D103
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):  # noqa: D103
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
