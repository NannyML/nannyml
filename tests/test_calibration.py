#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit tests for the calibration module."""
import numpy as np
import pandas as pd
import pytest

from nannyml.calibration import (
    IsotonicCalibrator,
    _get_bin_edges,
    _calculate_expected_calibration_error,
    needs_calibration,
)
from nannyml.exceptions import InvalidArgumentsException


@pytest.mark.parametrize('vector,bin_count', [([], 0), ([], 1), ([0], 1), (np.arange(2), 1), (np.ones(3), 5)])
def test_get_bin_edges_raises_invalid_arguments_exception_when_given_too_few_samples(vector, bin_count):  # noqa: D103
    with pytest.raises(InvalidArgumentsException):
        _ = _get_bin_edges(vector, bin_count)


@pytest.mark.parametrize('vector,bin_count', [(np.arange(0, 10), 'foo')])
def test_get_bin_edges_raises_invalid_arguments_exception_when_given_wrong_calibration_method(  # noqa: D103
    vector, bin_count
):
    with pytest.raises(ValueError):
        _ = _get_bin_edges(vector, bin_count)


@pytest.mark.parametrize(
    'vector,bin_count,edges',
    [
        (np.arange(0, 20), 4, [0, 4.75, 9.5, 14.25, 19.00000001]),
        (np.arange(0, 10), 3, [0, 3, 6, 9.00000001]),
        (np.arange(0, 10), 'auto', [0, 1.8, 3.6, 5.4, 7.2, 9.00000001]),
        (np.arange(0, 10), 'fd', [0, 3, 6, 9.00000001]),
        (np.arange(0, 10), 'doane', [0, 1.8, 3.6, 5.4, 7.2, 9.00000001]),
        (np.arange(0, 10), 'scott', [0, 4.5, 9.00000001]),
        (
            np.concatenate([np.ones(5), np.zeros(5)]),
            'stone',
            np.concatenate([np.linspace(0, 1, 101)[:-1], [1.00000001]]),
        ),
        (np.arange(0, 10), 'rice', [0, 1.8, 3.6, 5.4, 7.2, 9.00000001]),
        (np.arange(0, 10), 'sturges', [0, 1.8, 3.6, 5.4, 7.2, 9.00000001]),
        (np.arange(0, 10), 'sqrt', [0, 2.25, 4.5, 6.75, 9.00000001]),
    ],
)
def test_get_bin_edges_works_correctly(vector, bin_count, edges):  # noqa: D103
    sut = _get_bin_edges(vector, bin_count)

    assert len(sut) == len(edges)
    assert sorted(sut) == sorted(edges)


@pytest.mark.parametrize(
    'y_true,y_pred_proba,bin_edges,calibration_error',
    [
        (
            np.concatenate([np.ones(5), np.zeros(5)]),
            np.concatenate([np.ones(5), np.zeros(5)]),
            [0, 0.25, 0.5, 0.75, 1.00000001],
            0,
        ),
        (np.concatenate([np.ones(5), np.zeros(5)]), np.repeat(0, 10), [0, 0.25, 0.5, 0.75, 1.00000001], 0.5),
        (np.concatenate([np.ones(5), np.zeros(5)]), np.repeat(0.5, 10), [0, 0.25, 0.5, 0.75, 1.00000001], 0),
        (
            np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1]),
            np.array([0.1, 0.2, 0.45, 0.1, 0.4, 0.98, 0.24, 0.2, 0.39, 0.4]),
            [0, 0.25, 0.5, 0.75, 1.00000001],
            0.35,
        ),
    ],
)
def test_calculate_expected_calibration_error_works_correctly(
    y_true, y_pred_proba, bin_edges, calibration_error
):  # noqa: D103
    sut = _calculate_expected_calibration_error(y_true, y_pred_proba, bin_edges)

    assert sut == calibration_error


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
    y_pred_proba = pd.Series(np.asarray([0, 0, 0, np.NaN, 0, 0, 1, 1, 1, 1, 1, 1]))
    with pytest.raises(InvalidArgumentsException, match='predicted probabilities contain NaN.'):
        _ = needs_calibration(y_true, y_pred_proba, IsotonicCalibrator())


def test_needs_calibration_returns_false_when_roc_auc_score_equals_one():  # noqa: D103
    y_true = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred_proba = y_true
    sut = needs_calibration(y_true, y_pred_proba, IsotonicCalibrator())
    assert sut is False
