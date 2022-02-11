#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Calibrating model scores into probabilities."""
import abc
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedShuffleSplit

from nannyml import InvalidArgumentsException


class Calibrator(abc.ABC):
    """Class that is able to calibrate ``y_pred_proba`` scores into probabilities."""

    def calibrate(self, y_pred_proba: pd.Series):
        """Perform calibration of prediction scores.

        Parameters
        ----------
        y_pred_proba: pd.Series
            An array of prediction scores.

        """
        raise NotImplementedError


class IsotonicCalibrator(Calibrator):
    """Calibrates using IsotonicRegression model."""

    def __init__(self, y_pred_proba, y_true):
        """Creates a new IsotonicCalibrator."""
        regressor = IsotonicRegression(out_of_bounds="clip", increasing=True)
        regressor.fit(y_pred_proba, y_true)
        self._regressor = regressor

    def calibrate(self, y_pred_proba: pd.Series):
        """Perform calibration of prediction scores.

        Parameters
        ----------
        y_pred_proba: pd.Series
            An array of prediction scores.

        """
        return self._regressor.predict(y_pred_proba)


def _get_bin_index_edges(vector_length: int, bin_count: int) -> List[Tuple[int, int]]:
    if vector_length <= 2 * bin_count:
        bin_count = vector_length // 2
        if bin_count < 2:
            raise InvalidArgumentsException(
                "cannot split into minimum of 2 bins. Current sample size "
                f"is {vector_length}, please increase sample size. "
            )

    bin_width = vector_length // bin_count
    bin_edges = np.asarray(range(0, vector_length + 1, bin_width))
    bin_edges[-1] = vector_length
    bin_index_left = bin_edges[:-1]
    bin_index_right = bin_edges[1:]
    bin_index_edges = [(x, y) for x, y in zip(bin_index_left, bin_index_right)]
    return bin_index_edges


def _calculate_expected_calibration_error(
    y_true: pd.Series, y_pred_proba: pd.Series, bin_index_edges: List[Tuple[int, int]]
) -> Any:
    terms = []

    y_pred_proba, y_true = np.asarray(y_pred_proba), np.asarray(y_true)

    # sort both y_pred_proba and y_true, just to make sure
    sort_index = y_pred_proba.argsort()
    y_pred_proba = y_pred_proba[sort_index]
    y_true = y_true[sort_index]

    for left_edge, right_edge in bin_index_edges:
        bin_proba = y_pred_proba[left_edge, right_edge]
        bin_true = y_true[left_edge, right_edge]
        mean_bin_proba = np.mean(bin_proba)
        mean_bin_true = np.mean(bin_true)
        weight = len(bin_proba) / len(y_pred_proba)
        terms.append(weight * abs(mean_bin_proba - mean_bin_true))

    expected_calibration_error = np.sum(terms)
    return expected_calibration_error


def needs_calibration(
    calibrator: Calibrator, y_true: pd.Series, y_pred_proba: pd.Series, bin_count: int = 10, split_count: int = 3
) -> bool:
    """Returns whether a series of prediction scores benefits from additional calibration or not.

    Repeatedly splits the provided Series into binned train and test sets and checks if applying calibration using
    the provided ``Calibrator`` resulted in a lower Expected Calibration Error (ECE) for each bin.
    If this was true for each iteration then return True. If the ECE was raised even once then return False.

    Parameters
    ----------
    calibrator : Calibrator
        The Calibrator to use during testing.
    y_true : pd.Series
        Truth values.
    y_pred_proba :
        Prediction scores.
    bin_count : int
        Desired amount of bins to calculate ECE on.
    split_count : int
        Desired number of splits to make, i.e. number of times to evaluate calibration.

    Returns
    -------
    needs_calibration: bool
        True when the scores benefit from calibration, False otherwise.

    """
    sss = StratifiedShuffleSplit(n_splits=split_count, test_size=0.4, random_state=42)
    ece_diffs = []

    for train, test in sss.split(y_pred_proba, y_true):
        y_pred_proba_test, y_true_test = y_pred_proba[test], y_true[test]

        calibrated_y_pred_proba_test = calibrator.calibrate(y_pred_proba_test)

        bin_index_edges = _get_bin_index_edges(len(y_pred_proba_test), bin_count)
        ece_before_calibration = _calculate_expected_calibration_error(y_true_test, y_pred_proba_test, bin_index_edges)
        ece_after_calibration = _calculate_expected_calibration_error(
            y_true_test, calibrated_y_pred_proba_test, bin_index_edges
        )
        ece_diffs.append(ece_before_calibration - ece_after_calibration)

    if any(np.asarray(ece_diffs) < 0):
        return False
    else:
        return True
