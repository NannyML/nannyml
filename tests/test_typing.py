#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from typing import Tuple

import pandas as pd
import pytest

from nannyml._typing import UseCase, class_labels, derive_use_case
from nannyml.datasets import load_synthetic_multiclass_classification_dataset, load_synthetic_regression_dataset
from nannyml.exceptions import InvalidArgumentsException


@pytest.fixture
def regression_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_synthetic_regression_dataset()


@pytest.fixture
def multiclass_classification_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_synthetic_multiclass_classification_dataset()


@pytest.mark.parametrize(
    'y_pred_proba, use_case',
    [
        ('y_pred_proba', UseCase.CLASSIFICATION_BINARY),
        ({'class1': 'class1_col', 'class2': 'class2_col'}, UseCase.CLASSIFICATION_MULTICLASS),
        (None, UseCase.REGRESSION),
    ],
)
def test_derive_use_case(y_pred_proba, use_case):
    assert derive_use_case(y_pred_proba) == use_case


@pytest.mark.parametrize(
    'y_pred_proba, expected_labels',
    [({'C': 'col_c', 'A': 'col_a', 'B': 'col_b'}, ['A', 'B', 'C']), ({}, []), (None, [])],
)
def test_class_labels(y_pred_proba, expected_labels):
    labels = class_labels(y_pred_proba)

    assert labels == expected_labels


def test_class_labels_raises_invalid_args_exception_when_not_given_dict():
    with pytest.raises(InvalidArgumentsException):
        _ = class_labels('err')
