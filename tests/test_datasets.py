#  Author:  Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0


"""Testing the NannyML datasets.py functionality."""

import pytest

from nannyml.datasets import load_modified_california_housing_dataset, load_synthetic_sample


def test_runs_load_synthetic_sample():  # noqa: D103
    try:
        _ = load_synthetic_sample()
    except Exception:
        pytest.fail()


def test_runs_load_modified_california_housing_dataset():  # noqa: D103
    try:
        _ = load_modified_california_housing_dataset()
    except Exception:
        pytest.fail()
