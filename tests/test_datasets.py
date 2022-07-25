#  Author:  Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0


"""Testing the NannyML datasets.py functionality."""

import pytest

from nannyml.datasets import (
    load_modified_california_housing_dataset,
    load_synthetic_binary_classification_dataset,
    load_synthetic_car_loan_dataset,
    load_synthetic_multiclass_classification_dataset,
)


def test_runs_load_synthetic_binary_classification_dataset():  # noqa: D103
    try:
        _ = load_synthetic_binary_classification_dataset()
    except Exception:
        pytest.fail()


def test_runs_load_synthetic_multiclass_classification_dataset():  # noqa: D103
    try:
        _ = load_synthetic_multiclass_classification_dataset()
    except Exception:
        pytest.fail()


def test_runs_load_modified_california_housing_dataset():  # noqa: D103
    try:
        _ = load_modified_california_housing_dataset()
    except Exception:
        pytest.fail()


def test_runs_load_synthetic_car_loan_dataset():  # noqa: D103
    try:
        _ = load_synthetic_car_loan_dataset()
    except Exception:
        pytest.fail()
