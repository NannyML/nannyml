#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit tests for the calibration module."""
import numpy as np
import pandas as pd
import pytest

from nannyml.calibration import IsotonicCalibrator, _get_bin_index_edges, needs_calibration
from nannyml.exceptions import InvalidArgumentsException


@pytest.mark.parametrize('vector_size,bin_count', [(0, 0), (0, 1), (1, 1), (2, 1), (3, 5)])
def test_get_bin_edges_raises_invalid_arguments_exception_when_given_too_few_samples(  # noqa: D103
    vector_size, bin_count
):
    with pytest.raises(InvalidArgumentsException):
        _ = _get_bin_index_edges(vector_size, bin_count)


@pytest.mark.parametrize(
    'vector_length,bin_count,edges',
    [
        (20, 4, [(0, 5), (5, 10), (10, 15), (15, 20)]),
        (10, 3, [(0, 3), (3, 6), (6, 10)]),
    ],
)
def test_get_bin_edges_works_correctly(vector_length, bin_count, edges):  # noqa: D103
    sut = _get_bin_index_edges(vector_length, bin_count)

    assert len(sut) == len(edges)
    assert sorted(sut) == sorted(edges)


def test_needs_calibration_returns_false_when_calibration_does_not_always_improves_ece():  # noqa: D103
    y_true = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred_proba = y_true
    shuffled_indexes = np.random.permutation(len(y_true))
    y_true, y_pred_proba = y_true[shuffled_indexes], y_pred_proba[shuffled_indexes]
    sut = needs_calibration(y_true, y_pred_proba, IsotonicCalibrator(), bin_count=2, split_count=3)
    assert not sut


def test_needs_calibration_returns_true_when_calibration_always_improves_ece():  # noqa: D103
    y_true = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred_proba = abs(1 - y_true)
    shuffled_indexes = np.random.permutation(len(y_true))
    y_true, y_pred_proba = y_true[shuffled_indexes], y_pred_proba[shuffled_indexes]
    sut = needs_calibration(y_true, y_pred_proba, IsotonicCalibrator())
    assert sut


def test_needs_calibration_raises_invalid_args_exception_when_y_true_contains_nan():  # noqa: D103
    y_true = pd.Series([0, 0, 0, 0, 0, np.NaN, 1, 1, 1, 1, 1, 1])
    y_pred_proba = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    with pytest.raises(InvalidArgumentsException, match='target values contain NaN.'):
        _ = needs_calibration(y_true, y_pred_proba, IsotonicCalibrator())


def test_needs_calibration_raises_invalid_args_exception_when_y_pred_proba_contains_nan():  # noqa: D103
    y_true = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred_proba = np.asarray([0, 0, 0, np.NaN, 0, 0, 1, 1, 1, 1, 1, 1])
    with pytest.raises(InvalidArgumentsException, match='predicted probabilities contain NaN.'):
        _ = needs_calibration(y_true, y_pred_proba, IsotonicCalibrator())


def test_needs_calibration_returns_false_when_roc_auc_score_equals_one():  # noqa: D103
    y_true = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred_proba = y_true
    sut = needs_calibration(y_true, y_pred_proba, IsotonicCalibrator())
    assert sut is False
