#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from typing import Tuple

import pandas as pd
import pytest

from nannyml._typing import ProblemType, class_labels
from nannyml.datasets import load_synthetic_car_price_dataset, load_synthetic_multiclass_classification_dataset
from nannyml.exceptions import InvalidArgumentsException


@pytest.fixture
def regression_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_synthetic_car_price_dataset()


@pytest.fixture
def multiclass_classification_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_synthetic_multiclass_classification_dataset()


@pytest.mark.parametrize(
    'problem_type_str, problem_type',
    [
        ('classification_binary', ProblemType.CLASSIFICATION_BINARY),
        ('classification_multiclass', ProblemType.CLASSIFICATION_MULTICLASS),
        ('regression', ProblemType.REGRESSION),
    ],
)
def test_problem_type_parsing(problem_type_str, problem_type):
    assert ProblemType.parse(problem_type_str) == problem_type


def test_problem_type_parsing_raises_invalid_args_exc_when_given_unknown_problem_type_str():
    with pytest.raises(InvalidArgumentsException):
        _ = ProblemType.parse('foo')


@pytest.mark.parametrize(
    'y_pred_proba, expected_labels',
    [({'C': 'col_c', 'A': 'col_a', 'B': 'col_b'}, ['A', 'B', 'C']), ({}, [])],
)
def test_class_labels(y_pred_proba, expected_labels):
    labels = class_labels(y_pred_proba)

    assert labels == expected_labels


def test_class_labels_raises_invalid_args_exception_when_not_given_dict():
    with pytest.raises(InvalidArgumentsException):
        _ = class_labels('err')
