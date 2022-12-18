#  Author:   Niels Nuyttens  <niels@nannyml.com>
#            Jakub Bialek    <jakub@nannyml.com>
#
#  License: Apache Software License 2.0

"""Calibrating model scores into probabilities."""
import abc
from typing import Any, Callable, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
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


class CalibratorFactory:
    """Factory class to aid in construction of Calibrators."""

    _calibrators = {'isotonic': lambda args: IsotonicCalibrator()}

    @classmethod
    def register_calibrator(cls, key: str, create_calibrator: Callable):
        """Registers a new calibrator to the index.

        This index associates a certain key with a function that can be used to construct a new Calibrator instance.

        Parameters
        ----------
        key: str
            The key used to retrieve a Calibrator. When providing a key that is already in the index, the value
            will be overwritten.
        create_calibrator: Callable
            A function that - given a ``**kwargs`` argument - create a new instance of a Calibrator subclass.

        Examples
        --------
        >>> CalibratorFactory.register_calibrator('isotonic', lambda kwargs: IsotonicCalibrator())
        """
        cls._calibrators[key] = create_calibrator

    @classmethod
    def create(cls, key: Optional[str], **kwargs):
        """Creates a new Calibrator given a key value and optional keyword args.

        If the provided key equals ``None``, then a new instance of the default Calibrator (IsotonicCalibrator)
        will be returned.

        If a non-existent key is provided an ``InvalidArgumentsException`` is raised.

        Parameters
        ----------
        key : str
            The key used to retrieve a Calibrator. When providing a key that is already in the index, the value
            will be overwritten.
        kwargs : dict
            Optional keyword arguments that will be passed along to the function associated with the key.
            It can then use these arguments during the creation of a new Calibrator instance.

        Returns
        -------
        calibrator: Calibrator
            A new instance of a specific Calibrator subclass.

        Examples
        --------
        >>> calibrator = CalibratorFactory.create('isotonic', kwargs={'foo': 'bar'})
        """
        default = IsotonicCalibrator()
        if key is None:
            return default

        if key not in cls._calibrators:
            raise InvalidArgumentsException(
                f"calibrator {key} unknown. " f"Please provide one of the following: {cls._calibrators.keys()}"
            )

        return cls._calibrators.get(key, default)


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


class NoopCalibrator(Calibrator):
    """A Calibrator subclass that simply returns the inputs unaltered."""

    def fit(self, y_pred_proba: np.ndarray, y_true: np.ndarray):
        """Fit nothing and just return the calibrator."""
        return self

    def calibrate(self, y_pred_proba: np.ndarray):
        """Calibrate nothing and just return the original ``y_pred_proba`` inputs."""
        return np.asarray(y_pred_proba)


def _get_bin_edges(vector: np.ndarray, bin_count: Union[int, str]) -> np.ndarray:
    """Generates edges of bins for specified vector and number of bins required.

    Parameters
    ----------
    vector : np.ndarray
        The vector that will be binned using bins.
    bin_count : Union[int, str]
        If int, number of bins that will be generated.
        If str, it should be one of the methods to calculate the optimal bin width available in numpy.histogram:
        ``['auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt']``

    Returns
    -------
    bin_edges : list of bin edges
        See the example below for best intuition.

    Examples
    --------
    >>> get_bin_edge_indexes(20, 4)
    [0, 4.75, 9.5, 14.25, 19.00000001]

    """
    if isinstance(bin_count, int) and (bin_count <= 0):
        raise InvalidArgumentsException("`bin_count` must be positive, when an integer")

    bin_edges = np.histogram_bin_edges(vector, bin_count)
    # Add an epsilon to the last bin edge to make the final bin right-edge inclusive
    bin_edges[-1] += 1e-8

    if (len(bin_edges) == 2) or (len(np.unique(np.digitize(vector, bin_edges, 'right'))) < 2):  # pragma: no branch
        raise InvalidArgumentsException(
            "cannot split into minimum of 2 bins. Current number of bins "
            f"is {len(bin_edges)}, please increase number of bins or sample size. "
        )

    return bin_edges


def _calculate_expected_calibration_error(y_true: np.ndarray, y_pred_proba: np.ndarray, bin_edges: List[float]) -> Any:
    terms = []

    y_pred_proba, y_true = np.asarray(y_pred_proba), np.asarray(y_true)
    bins_proba = np.digitize(y_pred_proba, bin_edges, 'right')

    for bin in np.unique(bins_proba):
        bin_proba = y_pred_proba[np.where(bins_proba == bin)]
        bin_true = y_true[np.where(bins_proba == bin)]
        mean_bin_proba = np.mean(bin_proba)
        mean_bin_true = np.mean(bin_true)
        weight = len(bin_proba) / len(y_pred_proba)
        terms.append(weight * abs(mean_bin_proba - mean_bin_true))

    expected_calibration_error = np.sum(terms)
    return expected_calibration_error


def needs_calibration(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    calibrator: Calibrator,
    bin_count: Union[int, str] = 10,
    split_count: int = 10,
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
    y_true : np.array
        Series with reference binary targets - ``0`` or ``1``. Shape ``(n,)``.
    y_pred_proba : np.array
        Series or DataFrame of continuous reference scores/probabilities. Has to be the same shape as ``y_true``.
    bin_count : Union[int, str]
        If int, desired amount of bins to calculate ECE on.
        If str, it should be one of the methods to calculate the optimal bin width available in numpy.histogram:
        ``['auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt']``
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
    if y_true.dtype == 'object':
        if pd.isnull(y_true).any():
            raise InvalidArgumentsException(
                'target values contain NaN. ' 'Please ensure reference targets do not contain NaN values.'
            )
    else:
        if np.isnan(y_true).any():
            raise InvalidArgumentsException(
                'target values contain NaN. ' 'Please ensure reference targets do not contain NaN values.'
            )

    if np.isnan(y_pred_proba).any():
        raise InvalidArgumentsException(
            'predicted probabilities contain NaN. '
            'Please ensure reference predicted probabilities do not contain NaN values.'
        )

    if isinstance(bin_count, str) and (bin_count not in np.lib.histograms._hist_bin_selectors):
        raise InvalidArgumentsException(f'unknown method `{bin_count}` for bin_count given.')

    # Reset indices to deal with subsetting vs. index results from stratified shuffle split
    # y_pred_proba = y_pred_proba.reset_index(drop=True)
    # y_true = y_true.reset_index(drop=True)

    if roc_auc_score(y_true, y_pred_proba, multi_class='ovr') > 0.999:
        return False

    sss = StratifiedShuffleSplit(n_splits=split_count, test_size=0.1, random_state=42)

    list_y_true_test = []
    list_y_pred_proba_test = []
    list_calibrated_y_pred_proba_test = []

    for train, test in sss.split(y_pred_proba, y_true):
        if isinstance(y_pred_proba, pd.DataFrame):
            y_pred_proba_train, y_true_train = y_pred_proba.iloc[train, :], y_true[train]
            y_pred_proba_test, y_true_test = y_pred_proba.iloc[test, :], y_true[test]
        else:
            y_pred_proba_train, y_true_train = y_pred_proba[train], y_true[train]
            y_pred_proba_test, y_true_test = y_pred_proba[test], y_true[test]

        calibrator.fit(y_pred_proba_train, y_true_train)
        calibrated_y_pred_proba_test = calibrator.calibrate(y_pred_proba_test)

        list_y_true_test.append(y_true_test)
        list_y_pred_proba_test.append(y_pred_proba_test)
        list_calibrated_y_pred_proba_test.append(calibrated_y_pred_proba_test)

    vec_y_true_test = np.concatenate(list_y_true_test)
    vec_y_pred_proba_test = np.concatenate(list_y_pred_proba_test)
    vec_calibrated_y_pred_proba_test = np.concatenate(list_calibrated_y_pred_proba_test)

    bin_edges = _get_bin_edges(vec_y_pred_proba_test, bin_count)
    ece_before_calibration = _calculate_expected_calibration_error(vec_y_true_test, vec_y_pred_proba_test, bin_edges)
    ece_after_calibration = _calculate_expected_calibration_error(
        vec_y_true_test, vec_calibrated_y_pred_proba_test, bin_edges
    )

    return ece_before_calibration > ece_after_calibration
