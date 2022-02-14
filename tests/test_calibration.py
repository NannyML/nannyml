#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit tests for the calibration module."""
import numpy as np
import pytest

from nannyml.calibration import IsotonicCalibrator, _get_bin_index_edges, needs_calibration
from nannyml.exceptions import InvalidArgumentsException


@pytest.mark.parametrize('vector_size,bin_count', [(0, 0), (0, 1), (1, 1), (2, 1), (3, 5)])
def test_get_bin_edges_raises_invalid_arguments_exception_when_given_too_few_samples(
    vector_size, bin_count  # noqa: D103
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
    y_true = np.asarray([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1])

    y_pred_proba = np.asarray([0.01, 0.02, 0.03, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.7, 0.7])

    calibrator = IsotonicCalibrator()
    sut = needs_calibration(calibrator, y_true, y_pred_proba, bin_count=2, split_count=3)
    assert not sut


def test_needs_calibration_returns_true_when_calibration_always_improves_ece():  # noqa: D103
    y_true = np.asarray([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1])

    y_pred_proba = np.asarray([0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 52])

    calibrator = IsotonicCalibrator()
    sut = needs_calibration(calibrator, y_true, y_pred_proba, bin_count=2, split_count=3)
    assert sut


def test_needs_calibration_returns_false_when_calibration_sometimes_improves_ece():  # noqa: D103
    y_true = np.asarray([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1])

    y_pred_proba = np.asarray([0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.7, 0.7, 0.7, 0.7])

    calibrator = IsotonicCalibrator()
    sut = needs_calibration(calibrator, y_true, y_pred_proba, bin_count=2, split_count=3)
    assert not sut
