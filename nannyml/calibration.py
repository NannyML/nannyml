#  Author:   Niels Nuyttens  <niels@nannyml.com>
#            Jakub Bialek    <jakub@nannyml.com>
#
#  License: Apache Software License 2.0

"""Calibrating model scores into probabilities."""
import abc
from typing import Any, List, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedShuffleSplit

from nannyml.exceptions import InvalidArgumentsException


class Calibrator(abc.ABC):
    """Class that is able to calibrate ``y_pred_proba`` scores into probabilities."""

    def fit(self, y_pred_proba: np.ndarray, y_true: np.ndarray):
        """Fits the calibrator using a reference data set.

        Parameters
        ----------
        y_pred_proba: numpy.ndarray
            Vector of continuous reference scores/probabilities. Has to be the same shape as y_true.
        y_true : numpy.ndarray
            Vector with reference binary targets - 0 or 1. Shape (n,).
        """
        raise NotImplementedError

    def calibrate(self, y_pred_proba: np.ndarray):
        """Perform calibration of prediction scores.

        Parameters
        ----------
        y_pred_proba: numpy.ndarray
            Vector of continuous scores/probabilities. Has to be the same shape as y_true.
        """
        raise NotImplementedError


class IsotonicCalibrator(Calibrator):
    """Calibrates using IsotonicRegression model."""

    def __init__(self):
        """Creates a new IsotonicCalibrator."""
        regressor = IsotonicRegression(out_of_bounds="clip", increasing=True)
        self._regressor = regressor

    def fit(self, y_pred_proba: np.ndarray, y_true: np.ndarray):
        """Fits the calibrator using a reference data set.

        Parameters
        ----------
        y_pred_proba: numpy.ndarray
            Vector of continuous reference scores/probabilities. Has to be the same shape as y_true.
        y_true : numpy.ndarray
            Vector with reference binary targets - 0 or 1. Shape (n,).
        """
        self._regressor.fit(y_pred_proba, y_true)

    def calibrate(self, y_pred_proba: np.ndarray):
        """Perform calibration of prediction scores.

        Parameters
        ----------
        y_pred_proba: numpy.ndarray
            Vector of continuous scores/probabilities. Has to be the same shape as ``y_true``.
        """
        return self._regressor.predict(y_pred_proba)


def _get_bin_index_edges(vector_length: int, bin_count: int) -> List[Tuple[int, int]]:
    """Generates edges of bins for specified vector length and number of bins required.

    Parameters
    ----------
    vector_length : int
        The length of the vector that will be binned using bins.
    bin_count : int
        Number of bins and bin edges that will be generated.

    Returns
    -------
    bin_index_edges : list of tuples with bin edges (indexes)
        See the example below for best intuition.

    Examples
    --------
    >>> get_bin_edge_indexes(20, 4)
    [(0, 5), (5, 10), (10, 15), (15, 20)]

    """
    if vector_length <= 2 * bin_count:
        bin_count = vector_length // 2
        if bin_count < 2:  # pragma: no branch
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
    y_true: np.ndarray, y_pred_proba: np.ndarray, bin_index_edges: List[Tuple[int, int]]
) -> Any:
    terms = []

    y_pred_proba, y_true = np.asarray(y_pred_proba), np.asarray(y_true)

    # sort both y_pred_proba and y_true, just to make sure
    sort_index = y_pred_proba.argsort()
    y_pred_proba = y_pred_proba[sort_index]
    y_true = y_true[sort_index]

    for left_edge, right_edge in bin_index_edges:
        bin_proba = y_pred_proba[left_edge:right_edge]
        bin_true = y_true[left_edge:right_edge]
        mean_bin_proba = np.mean(bin_proba)
        mean_bin_true = np.mean(bin_true)
        weight = len(bin_proba) / len(y_pred_proba)
        terms.append(weight * abs(mean_bin_proba - mean_bin_true))

    expected_calibration_error = np.sum(terms)
    return expected_calibration_error


def needs_calibration(
    y_true: np.ndarray, y_pred_proba: np.ndarray, calibrator: Calibrator, bin_count: int, split_count: int
) -> bool:
    """Returns whether a series of prediction scores benefits from additional calibration or not.

    Performs probability calibration in cross validation loop. For each fold a difference
    of Expected Calibration Error (ECE) between non calibrated and calibrated
    probabilites is calculated. If in any of the folds the difference is lower than zero
    (i.e. ECE of calibrated probability is larger than that of non-calibrated) returns ``False``.
    Otherwise - returns ``True``.

    Parameters
    ----------
    calibrator : Calibrator
        The Calibrator to use during testing.
    y_true : numpy.ndarray
        Vector with reference binary targets - ``0`` or ``1``. Shape ``(n,)``.
    y_pred_proba :
        Vector of continuous reference scores/probabilities. Has to be the same shape as ``y_true``.
    bin_count : int
        Desired amount of bins to calculate ECE on.
    split_count : int
        Desired number of splits to make, i.e. number of times to evaluate calibration.

    Returns
    -------
    needs_calibration: bool
        ``True`` when the scores benefit from calibration, ``False`` otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> from nannyml.calibration import IsotonicCalibrator
    >>> np.random.seed(1)
    >>> y_true = np.random.binomial(1, 0.5, 10)
    >>> y_pred_proba = np.linspace(0, 1, 10)
    >>> calibrator = IsotonicCalibrator()
    >>> needs_calibration(y_true, y_pred_proba, calibrator, bin_count=2, split_count=3)
    True
    """
    sss = StratifiedShuffleSplit(n_splits=split_count, test_size=0.4, random_state=42)
    ece_diffs = []

    for train, test in sss.split(y_pred_proba, y_true):
        y_pred_proba_train, y_true_train = y_pred_proba[train], y_true[train]
        y_pred_proba_test, y_true_test = y_pred_proba[test], y_true[test]
        calibrator.fit(y_pred_proba_train, y_true_train)
        calibrated_y_pred_proba_test = calibrator.calibrate(y_pred_proba_test)

        bin_index_edges = _get_bin_index_edges(len(y_pred_proba_test), bin_count)
        ece_before_calibration = _calculate_expected_calibration_error(y_true_test, y_pred_proba_test, bin_index_edges)
        ece_after_calibration = _calculate_expected_calibration_error(
            y_true_test, calibrated_y_pred_proba_test, bin_index_edges
        )
        ece_diffs.append(ece_before_calibration - ece_after_calibration)

    if any(np.asarray(ece_diffs) <= 0):
        return False
    else:
        return True
