#  Author:   Niels Nuyttens  <niels@nannyml.com>
#            Jakub Bialek    <jakub@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing functions to estimate sampling error for binary classification metrics.

The implementation of the sampling error estimation is split into two functions.

The first function is called during fitting and will calculate the sampling error components based the reference data.
Most of the time these will be the standard deviation of the distribution of differences between
``y_true`` and ``y_pred`` and the fraction of positive labels in ``y_true``.

The second function will be called during calculation or estimation. It takes the predetermined error components and
combines them with the size of the (analysis) data to give an estimate for the sampling error.
"""

from typing import Tuple

import numpy as np
import pandas as pd


def _universal_sampling_error(reference_std, reference_fraction, data):
    return reference_std / np.sqrt(len(data) * reference_fraction)


def auroc_sampling_error_components(y_true_reference: pd.Series, y_pred_proba_reference: pd.Series) -> Tuple:
    """
    Estimation of AUROC sampling error. Calculation is based on the Variance Sum Law and expressing AUROC as
    a Mann-Whitney U statistic.

    Parameters
    ----------
    y_true_reference: pd.Series
        Target values for the reference dataset.
    y_pred_proba_reference: pd.Series
        Prediction values for the reference dataset.

    Returns
    -------
    (std, fraction): Tuple[np.ndarray, float]
    """

    y_true = y_true_reference.copy().reset_index(drop=True)
    y_pred_proba = y_pred_proba_reference.copy().reset_index(drop=True)

    if np.mean(y_true) > 0.5:
        y_true = abs(np.asarray(y_true) - 1)
        y_pred_proba = 1 - y_pred_proba

    sorted_idx = np.argsort(y_pred_proba)
    y_pred_proba = y_pred_proba[sorted_idx]
    y_true = y_true[sorted_idx]
    rank_order = np.asarray(range(len(y_pred_proba)))
    positive_ranks = y_true * rank_order
    indexes = np.unique(positive_ranks)[1:]
    ser = []
    for i, index in enumerate(indexes):
        ser.append(index - i)

    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    ser_multi = ser / n_neg
    fraction = n_pos / len(y_true)

    return np.std(ser_multi), fraction


def auroc_sampling_error(sampling_error_components, data):
    """
    Calculate the AUROC sampling error for a chunk of data.

    Parameters
    ----------
    sampling_error_components : a set of parameters that were derived from reference data.
    data : the (analysis) data you want to calculate or estimate a metric for.

    Returns
    -------
    sampling_error: float

    """
    reference_std, reference_fraction = sampling_error_components
    return _universal_sampling_error(reference_std, reference_fraction, data)


def f1_sampling_error_components(y_true_reference: pd.Series, y_pred_reference: pd.Series) -> Tuple:
    """
    Estimate sampling error of F1 using modified standard error of mean formula.

    Parameters
    ----------
    y_true_reference: pd.Series
        Target values for the reference dataset.
    y_pred_reference: pd.Series
        Predictions for the reference dataset.

    Returns
    -------
    (std, fraction): Tuple[np.ndarray, float]
    """
    TP = np.where((y_true_reference == y_pred_reference) & (y_pred_reference == 1), 1, np.nan)

    FP = np.where((y_true_reference != y_pred_reference) & (y_pred_reference == 1), 0, np.nan)
    FN = np.where((y_true_reference != y_pred_reference) & (y_pred_reference == 0), 0, np.nan)

    TP = TP[~np.isnan(TP)]

    FN = FN[~np.isnan(FN)]
    FP = FP[~np.isnan(FP)]

    tp_fp_fn = np.concatenate([TP, FN, FP])

    correcting_factor = len(tp_fp_fn) / ((len(FN) + len(FP)) * 0.5 + len(TP))
    obs_level_f1 = tp_fp_fn * correcting_factor
    fraction_of_relevant = len(tp_fp_fn) / len(y_pred_reference)

    return np.std(obs_level_f1), fraction_of_relevant


def f1_sampling_error(sampling_error_components, data):
    """
    Calculate the F1 sampling error for a chunk of data.

    Parameters
    ----------
    sampling_error_components : a set of parameters that were derived from reference data.
    data : the (analysis) data you want to calculate or estimate a metric for.

    Returns
    -------
    sampling_error: float

    """
    reference_std, reference_fraction = sampling_error_components
    return _universal_sampling_error(reference_std, reference_fraction, data)


def precision_sampling_error_components(y_true_reference: pd.Series, y_pred_reference: pd.Series) -> Tuple:
    """
    Estimate sampling error for precision using modified standard error of mean formula.

    Parameters
    ----------
    y_true_reference: pd.Series
        Target values for the reference dataset.
    y_pred_reference: pd.Series
        Predictions for the reference dataset.

    Returns
    -------
    (std, fraction): Tuple[np.ndarray, float]
    """
    TP = np.where((y_true_reference == y_pred_reference) & (y_pred_reference == 1), 1, np.nan)
    FP = np.where((y_true_reference != y_pred_reference) & (y_pred_reference == 1), 0, np.nan)

    TP = TP[~np.isnan(TP)]
    FP = FP[~np.isnan(FP)]
    obs_level_precision = np.concatenate([TP, FP])
    amount_positive_pred = np.sum(y_pred_reference)
    fraction_of_pos_pred = amount_positive_pred / len(y_pred_reference)

    return np.std(obs_level_precision), fraction_of_pos_pred


def precision_sampling_error(sampling_error_components, data):
    """
    Calculate the precision sampling error for a chunk of data.

    Parameters
    ----------
    sampling_error_components : a set of parameters that were derived from reference data.
    data : the (analysis) data you want to calculate or estimate a metric for.

    Returns
    -------
    sampling_error: float

    """
    reference_std, reference_fraction = sampling_error_components
    return _universal_sampling_error(reference_std, reference_fraction, data)


def recall_sampling_error_components(y_true_reference: pd.Series, y_pred_reference: pd.Series) -> Tuple:
    """
    Estimate sampling error for recall using modified standard error of mean formula.

    Parameters
    ----------
    y_true_reference: pd.Series
        Target values for the reference dataset.
    y_pred_reference: pd.Series
        Predictions for the reference dataset.

    Returns
    -------
    (std, fraction): Tuple[np.ndarray, float]
    """
    TP = np.where((y_true_reference == y_pred_reference) & (y_pred_reference == 1), 1, np.nan)
    FN = np.where((y_true_reference != y_pred_reference) & (y_pred_reference == 0), 0, np.nan)

    TP = TP[~np.isnan(TP)]
    FN = FN[~np.isnan(FN)]
    obs_level_recall = np.concatenate([TP, FN])
    fraction_of_relevant = sum(y_true_reference) / len(y_pred_reference)

    return np.std(obs_level_recall), fraction_of_relevant


def recall_sampling_error(sampling_error_components, data):
    """
    Calculate the recall sampling error for a chunk of data.

    Parameters
    ----------
    sampling_error_components : a set of parameters that were derived from reference data.
    data : the (analysis) data you want to calculate or estimate a metric for.

    Returns
    -------
    sampling_error: float

    """
    reference_std, reference_fraction = sampling_error_components
    return _universal_sampling_error(reference_std, reference_fraction, data)


def specificity_sampling_error_components(y_true_reference: pd.Series, y_pred_reference: pd.Series) -> Tuple:
    """
    Estimate sampling error for specificity using modified standard error of mean formula.

    Parameters
    ----------
    y_true_reference: pd.Series
        Target values for the reference dataset.
    y_pred_reference: pd.Series
        Predictions for the reference dataset.

    Returns
    -------
    (std, fraction): Tuple[np.ndarray, float]
    """
    TN = np.where((y_true_reference == y_pred_reference) & (y_pred_reference == 0), 1, np.nan)
    FP = np.where((y_true_reference != y_pred_reference) & (y_pred_reference == 1), 0, np.nan)

    TN = TN[~np.isnan(TN)]
    FP = FP[~np.isnan(FP)]
    obs_level_specificity = np.concatenate([TN, FP])
    fraction_of_relevant = len(obs_level_specificity) / len(y_pred_reference)

    return np.std(obs_level_specificity), fraction_of_relevant


def specificity_sampling_error(sampling_error_components, data):
    """
    Calculate the specificity sampling error for a chunk of data.

    Parameters
    ----------
    sampling_error_components : a set of parameters that were derived from reference data.
    data : the (analysis) data you want to calculate or estimate a metric for.

    Returns
    -------
    sampling_error: float

    """
    reference_std, reference_fraction = sampling_error_components
    return _universal_sampling_error(reference_std, reference_fraction, data)


def accuracy_sampling_error_components(y_true_reference: pd.Series, y_pred_reference: pd.Series) -> Tuple:
    """
    Estimate sampling error for accuracy.

    Parameters
    ----------
    y_true_reference: pd.Series
        Target values for the reference dataset.
    y_pred_reference: pd.Series
        Predictions for the reference dataset.

    Returns
    -------
    (std,): Tuple[np.ndarray]
    """
    y_true_reference = np.asarray(y_true_reference).astype(int)
    y_pred_reference = np.asarray(y_pred_reference).astype(int)

    correct_table = (y_true_reference == y_pred_reference).astype(int)

    return (np.std(correct_table),)


def accuracy_sampling_error(sampling_error_components: Tuple, data) -> float:
    """
    Calculate the accuracy sampling error for a chunk of data.

    Parameters
    ----------
    sampling_error_components : a set of parameters that were derived from reference data.
    data : the (analysis) data you want to calculate or estimate a metric for.

    Returns
    -------
    sampling_error: float

    """
    (reference_std,) = sampling_error_components
    return reference_std / np.sqrt(len(data))
