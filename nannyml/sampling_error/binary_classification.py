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

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from nannyml.exceptions import InvalidArgumentsException

# How many experiments to perform when doing resampling to approximate sampling error.
N_EXPERIMENTS = 50
# Max resample size - we don't need full reference if it is too big.
MAX_RESAMPLE_SIZE = 50_000


def _universal_sampling_error(reference_std, reference_fraction, data):
    return reference_std / np.sqrt(len(data) * reference_fraction)


def auroc_sampling_error_components(y_true_reference: pd.Series, y_pred_proba_reference: pd.Series) -> Tuple:
    """
    Calculate sampling error components for AUROC using reference data.
    Calculation is based on the Variance Sum Law and expressing AUROC as a Mann-Whitney U statistic.

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
    # keep converting to numpy here for now
    y_true = y_true_reference.to_numpy()
    y_pred_proba = y_pred_proba_reference.to_numpy()

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
    ser_multi = np.true_divide(ser, n_neg)
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


def ap_sampling_error_components(
    y_true_reference: pd.Series, y_pred_proba_reference: pd.Series
) -> Tuple[np.ndarray, int]:
    """
    Calculate sampling error components for AP using reference data.
    Calculation is done by calculating the sampling error on reference data and extrapolating
    for different sizes using 1/sqrt(n) approximation.

    Parameters
    ----------
    y_true_reference: pd.Series
        Target values for the reference dataset.
    y_pred_proba_reference: pd.Series
        Prediction values for the reference dataset.

    Returns
    -------
    (std, sample_size): Tuple[np.ndarray, int]
        Note that the sampling error component are different than usual!
    """

    # we don't need all reference if it's big (save compute)
    sample_size = np.minimum(y_true_reference.shape[0] // 2, MAX_RESAMPLE_SIZE)

    y_true_reference = y_true_reference.to_numpy()
    y_pred_proba_reference = y_pred_proba_reference.to_numpy()

    ap_results = []
    for _ in range(N_EXPERIMENTS):
        _indexes_for_sample = np.random.choice(y_true_reference.shape[0], sample_size, replace=True)
        sample_y_true_reference = y_true_reference[_indexes_for_sample]
        sample_y_pred_proba_reference = y_pred_proba_reference[_indexes_for_sample]
        ap_results.append(average_precision_score(sample_y_true_reference, sample_y_pred_proba_reference))
    return np.std(ap_results), sample_size


def ap_sampling_error(sampling_error_components, data):
    """
    Calculate the AUROC sampling error for a chunk of data.

    if first component is NaN (due to data quality) result will be nan

    Parameters
    ----------
    sampling_error_components : a set of parameters that were derived from reference data.
    data : the (analysis) data you want to calculate or estimate a metric for.

    Returns
    -------
    sampling_error: float

    """
    reference_std, sample_size = sampling_error_components
    analysis_size = data.shape[0]
    return reference_std * np.sqrt(sample_size / analysis_size)


def f1_sampling_error_components(y_true_reference: pd.Series, y_pred_reference: pd.Series) -> Tuple:
    """
    Calculate sampling error components for F1 using reference data.
    Calculation is based on modified standard error of mean formula.

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

    # If there's no true positives, false negatives or false positives, sampling error is NaN
    if tp_fp_fn.size == 0:
        return np.nan, 0

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
    Calculate sampling error components for precision using reference data.
    Calculation is based on modified standard error of mean formula.

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
    Calculate sampling error components for recall using reference data.
    Calculation is based on modified standard error of mean formula.

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
    Calculate sampling error components for specificity using reference data.
    Calculation is based on modified standard error of mean formula.

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
    Calculate sampling error components for accuracy using reference data.
    Calculation is based on modified standard error of mean formula.

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


def true_positive_sampling_error_components(
    y_true_reference: pd.Series, y_pred_reference: pd.Series, normalize_confusion_matrix: Union[str, None]
) -> Tuple[float, float, Union[str, None]]:
    """
    Estimate sampling error components for true positive rate using reference data.
    Calculation is based on modified standard error of mean formula.

    Parameters
    ----------
    y_true_reference: pd.Series
        Target values for the reference dataset.
    y_pred_reference: pd.Series
        Predictions for the reference dataset.
    normalize_confusion_matrix: str
        The type of normalization to apply to the confusion matrix.

    Returns
    -------
    (std, relevant_proportion, norm_type): Tuple[float, float, str]
    """
    y_true_reference = np.asarray(y_true_reference).astype(int)
    y_pred_reference = np.asarray(y_pred_reference).astype(int)

    obs_level_tp = np.where((y_true_reference == y_pred_reference) & (y_pred_reference == 1), 1, 0)

    obs_level_fn = np.where((y_true_reference != y_pred_reference) & (y_pred_reference == 0), 1, 0)

    obs_level_fp = np.where((y_true_reference != y_pred_reference) & (y_pred_reference == 1), 1, 0)

    num_tp = np.sum(obs_level_tp)
    num_fn = np.sum(obs_level_fn)
    num_fp = np.sum(obs_level_fp)

    if normalize_confusion_matrix is None:
        std = np.std(obs_level_tp)

        relevant_proportion = 1

    elif normalize_confusion_matrix == "all":
        std = np.std(obs_level_tp)

        relevant_proportion = 1

    elif normalize_confusion_matrix == "true":
        number_of_real_positives = num_fn + num_tp
        proportion_of_real_positives = number_of_real_positives / len(y_true_reference)

        obs_level_tp = np.concatenate([np.ones(num_tp), np.zeros(number_of_real_positives - num_tp)])

        std = np.std(obs_level_tp)

        relevant_proportion = proportion_of_real_positives

    elif normalize_confusion_matrix == "pred":
        number_of_pred_positives = num_fp + num_tp
        proportion_of_pred_positives = number_of_pred_positives / len(y_true_reference)

        obs_level_tp = np.concatenate([np.ones(num_tp), np.zeros(number_of_pred_positives - num_tp)])

        std = np.std(obs_level_tp)

        relevant_proportion = proportion_of_pred_positives

    else:
        raise InvalidArgumentsException(
            f"'normalize_confusion_matrix' should be None, 'true', 'pred' or 'all' "
            f"but got '{normalize_confusion_matrix}"
        )

    return std, relevant_proportion, normalize_confusion_matrix


def true_positive_sampling_error(sampling_error_components: Tuple, data) -> float:
    """
    Calculate the true positive rate sampling error for a chunk of data.

    Parameters
    ----------
    sampling_error_components : a set of parameters that were derived from reference data.
    data : the (analysis) data you want to calculate or estimate a metric for.

    Returns
    -------
    sampling_error: float
    """
    (reference_std, relevant_proportion, norm_type) = sampling_error_components

    if norm_type is None:
        tp_standard_error = (reference_std / np.sqrt(len(data))) * len(data)

    elif norm_type == "all":
        tp_standard_error = reference_std / np.sqrt(len(data))

    elif norm_type == "true" or norm_type == "pred":
        tp_standard_error = reference_std / np.sqrt(len(data) * relevant_proportion)

    else:
        raise InvalidArgumentsException(
            f"'normalize_confusion_matrix' should be None, 'true', 'pred' or 'all' " f"but got '{norm_type}"
        )

    return tp_standard_error


def true_negative_sampling_error_components(
    y_true_reference: pd.Series, y_pred_reference: pd.Series, normalize_confusion_matrix: Union[str, None]
) -> Tuple[float, float, Union[str, None]]:
    """
    Estimate sampling error components for true negative rate using reference data.
    Calculation is based on modified standard error of mean formula.

    Parameters
    ----------
    y_true_reference: pd.Series
        Target values for the reference dataset.
    y_pred_reference: pd.Series
        Predictions for the reference dataset.
    normalize_confusion_matrix: str
        The type of normalization to apply to the confusion matrix.

    Returns
    -------
    (std, relevant_proportion, norm_type): Tuple[float, float, str]
    """
    y_true_reference = np.asarray(y_true_reference).astype(int)
    y_pred_reference = np.asarray(y_pred_reference).astype(int)

    obs_level_tn = np.where((y_true_reference == y_pred_reference) & (y_pred_reference == 0), 1, 0)

    obs_level_fn = np.where((y_true_reference != y_pred_reference) & (y_pred_reference == 0), 1, 0)

    obs_level_fp = np.where((y_true_reference != y_pred_reference) & (y_pred_reference == 1), 1, 0)

    num_tn = np.sum(obs_level_tn)
    num_fn = np.sum(obs_level_fn)
    num_fp = np.sum(obs_level_fp)

    if normalize_confusion_matrix is None:
        std = np.std(obs_level_tn)

        relevant_proportion = 1

    elif normalize_confusion_matrix == "all":
        std = np.std(obs_level_tn)

        relevant_proportion = 1

    elif normalize_confusion_matrix == "true":
        number_of_real_negatives = num_fp + num_tn
        proportion_of_real_negatives = number_of_real_negatives / len(y_true_reference)

        obs_level_tn = np.concatenate([np.ones(num_tn), np.zeros(number_of_real_negatives - num_tn)])

        std = np.std(obs_level_tn)

        relevant_proportion = proportion_of_real_negatives

    elif normalize_confusion_matrix == "pred":
        number_of_pred_negatives = num_fn + num_tn
        proportion_of_pred_negatives = number_of_pred_negatives / len(y_true_reference)

        obs_level_tn = np.concatenate([np.ones(num_tn), np.zeros(number_of_pred_negatives - num_tn)])

        std = np.std(obs_level_tn)

        relevant_proportion = proportion_of_pred_negatives

    else:
        raise InvalidArgumentsException(
            f"'normalize_confusion_matrix' should be None, 'true', 'pred' or 'all' "
            f"but got '{normalize_confusion_matrix}"
        )

    return std, relevant_proportion, normalize_confusion_matrix


def true_negative_sampling_error(sampling_error_components: Tuple, data) -> float:
    """
    Calculate the true negative rate sampling error for a chunk of data.

    Parameters
    ----------
    sampling_error_components : a set of parameters that were derived from reference data.
    data : the (analysis) data you want to calculate or estimate a metric for.

    Returns
    -------
    sampling_error: float
    """
    (reference_std, relevant_proportion, norm_type) = sampling_error_components

    if norm_type is None:
        tn_standard_error = (reference_std / np.sqrt(len(data))) * len(data)

    elif norm_type == "all":
        tn_standard_error = reference_std / np.sqrt(len(data))

    elif norm_type == "true" or norm_type == "pred":
        tn_standard_error = reference_std / np.sqrt(len(data) * relevant_proportion)

    else:
        raise InvalidArgumentsException(
            f"'normalize_confusion_matrix' should be None, 'true', 'pred' or 'all' " f"but got '{norm_type}"
        )

    return tn_standard_error


def false_positive_sampling_error_components(
    y_true_reference: pd.Series, y_pred_reference: pd.Series, normalize_confusion_matrix: Union[str, None]
) -> Tuple[float, float, Union[str, None]]:
    """
    Estimate sampling error components for false positive rate using reference data.
    Calculation is based on modified standard error of mean formula.

    Parameters
    ----------
    y_true_reference: pd.Series
        Target values for the reference dataset.
    y_pred_reference: pd.Series
        Predictions for the reference dataset.
    normalize_confusion_matrix: str
        The type of normalization to apply to the confusion matrix.

    Returns
    -------
    (std, relevant_proportion, norm_type): Tuple[float, float, str]
    """
    y_true_reference = np.asarray(y_true_reference).astype(int)
    y_pred_reference = np.asarray(y_pred_reference).astype(int)

    obs_level_tp = np.where((y_true_reference == y_pred_reference) & (y_pred_reference == 1), 1, 0)

    obs_level_tn = np.where((y_true_reference == y_pred_reference) & (y_pred_reference == 0), 1, 0)

    obs_level_fp = np.where((y_true_reference != y_pred_reference) & (y_pred_reference == 1), 1, 0)

    num_tp = np.sum(obs_level_tp)
    num_tn = np.sum(obs_level_tn)
    num_fp = np.sum(obs_level_fp)

    if normalize_confusion_matrix is None:
        std = np.std(obs_level_fp)

        relevant_proportion = 1

    elif normalize_confusion_matrix == "all":
        std = np.std(obs_level_fp)

        relevant_proportion = 1

    elif normalize_confusion_matrix == "true":
        number_of_real_negatives = num_fp + num_tn
        proportion_of_real_negatives = number_of_real_negatives / len(y_true_reference)

        obs_level_fp = np.concatenate([np.ones(num_fp), np.zeros(number_of_real_negatives - num_fp)])

        std = np.std(obs_level_fp)

        relevant_proportion = proportion_of_real_negatives

    elif normalize_confusion_matrix == "pred":
        number_of_pred_positives = num_fp + num_tp
        proportion_of_pred_positives = number_of_pred_positives / len(y_true_reference)

        obs_level_fp = np.concatenate([np.ones(num_fp), np.zeros(number_of_pred_positives - num_fp)])

        std = np.std(obs_level_fp)

        relevant_proportion = proportion_of_pred_positives

    else:
        raise InvalidArgumentsException(
            f"'normalize_confusion_matrix' should be None, 'true', 'pred' or 'all' "
            f"but got '{normalize_confusion_matrix}"
        )

    return std, relevant_proportion, normalize_confusion_matrix


def false_positive_sampling_error(sampling_error_components: Tuple, data) -> float:
    """
    Calculate the false positive rate sampling error for a chunk of data.

    Parameters
    ----------
    sampling_error_components : a set of parameters that were derived from reference data.
    data : the (analysis) data you want to calculate or estimate a metric for.

    Returns
    -------
    sampling_error: float
    """
    (reference_std, relevant_proportion, norm_type) = sampling_error_components

    if norm_type is None:
        fp_standard_error = (reference_std / np.sqrt(len(data))) * len(data)

    elif norm_type == "all":
        fp_standard_error = reference_std / np.sqrt(len(data))

    elif norm_type == "true" or norm_type == "pred":
        fp_standard_error = reference_std / np.sqrt(len(data) * relevant_proportion)

    else:
        raise InvalidArgumentsException(
            f"'normalize_confusion_matrix' should be None, 'true', 'pred' or 'all' " f"but got '{norm_type}"
        )

    return fp_standard_error


def false_negative_sampling_error_components(
    y_true_reference: pd.Series, y_pred_reference: pd.Series, normalize_confusion_matrix: Union[str, None]
) -> Tuple[float, float, Union[str, None]]:
    """
    Estimate sampling error components for false negative rate using reference data.
    Calculation is based on modified standard error of mean formula.

    Parameters
    ----------
    y_true_reference: pd.Series
        Target values for the reference dataset.
    y_pred_reference: pd.Series
        Predictions for the reference dataset.
    normalize_confusion_matrix: str
        The type of normalization to apply to the confusion matrix.

    Returns
    -------
    (std, relevant_proportion, norm_type): Tuple[float, float, str]
    """
    y_true_reference = np.asarray(y_true_reference).astype(int)
    y_pred_reference = np.asarray(y_pred_reference).astype(int)

    obs_level_tp = np.where((y_true_reference == y_pred_reference) & (y_pred_reference == 1), 1, 0)

    obs_level_tn = np.where((y_true_reference == y_pred_reference) & (y_pred_reference == 0), 1, 0)

    obs_level_fn = np.where((y_true_reference != y_pred_reference) & (y_pred_reference == 0), 1, 0)

    num_tp = np.sum(obs_level_tp)
    num_tn = np.sum(obs_level_tn)
    num_fn = np.sum(obs_level_fn)

    if normalize_confusion_matrix is None:
        std = np.std(obs_level_fn)

        relevant_proportion = 1

    elif normalize_confusion_matrix == "all":
        std = np.std(obs_level_fn)

        relevant_proportion = 1  # Could be None, None

    elif normalize_confusion_matrix == "true":
        number_of_real_positives = num_fn + num_tp
        proportion_of_real_positives = number_of_real_positives / len(y_true_reference)

        obs_level_fn = np.concatenate([np.ones(num_fn), np.zeros(number_of_real_positives - num_fn)])

        std = np.std(obs_level_fn)

        relevant_proportion = proportion_of_real_positives

    elif normalize_confusion_matrix == "pred":
        number_of_pred_negatives = num_fn + num_tn
        proportion_of_pred_negatives = number_of_pred_negatives / len(y_true_reference)

        obs_level_fn = np.concatenate([np.ones(num_fn), np.zeros(number_of_pred_negatives - num_fn)])

        std = np.std(obs_level_fn)

        relevant_proportion = proportion_of_pred_negatives

    else:
        raise InvalidArgumentsException(
            f"'normalize_confusion_matrix' should be None, 'true', 'pred' or 'all' "
            f"but got '{normalize_confusion_matrix}"
        )

    return std, relevant_proportion, normalize_confusion_matrix


def false_negative_sampling_error(sampling_error_components: Tuple, data) -> float:
    """
    Calculate the false positive rate sampling error for a chunk of data.

    Parameters
    ----------
    sampling_error_components : a set of parameters that were derived from reference data.
    data : the (analysis) data you want to calculate or estimate a metric for.

    Returns
    -------
    sampling_error: float
    """
    (reference_std, relevant_proportion, norm_type) = sampling_error_components

    if norm_type is None:
        fn_standard_error = (reference_std / np.sqrt(len(data))) * len(data)

    elif norm_type == "all":
        fn_standard_error = reference_std / np.sqrt(len(data))

    elif norm_type == "true" or norm_type == "pred":
        fn_standard_error = reference_std / np.sqrt(len(data) * relevant_proportion)

    else:
        raise InvalidArgumentsException(
            f"'normalize_confusion_matrix' should be None, 'true', 'pred' or 'all' " f"but got '{norm_type}"
        )

    return fn_standard_error


def business_value_sampling_error_components(
    y_true_reference: pd.Series,
    y_pred_reference: pd.Series,
    business_value_matrix: np.ndarray,
    normalize_business_value: Optional[str],
) -> Tuple[float, Union[str, None]]:
    """
    Estimate sampling error for the false negative rate.
    Parameters
    ----------
    y_true_reference: pd.Series
        Target values for the reference dataset.
    y_pred_reference: pd.Series
        Predictions for the reference dataset.
    business_value_matrix: np.ndarray
        A 2x2 matrix of values for the business problem.
    normalize_business_value: Optional[str], default=None
            Determines how the business value will be normalized. Allowed values are None and 'per_prediction'.
    Returns
    -------
    components: tuple
    """
    y_true_reference = np.asarray(y_true_reference).astype(int)
    y_pred_reference = np.asarray(y_pred_reference).astype(int)

    obs_level_tp = np.where((y_true_reference == y_pred_reference) & (y_pred_reference == 1), 1, 0)
    obs_level_tn = np.where((y_true_reference == y_pred_reference) & (y_pred_reference == 0), 1, 0)
    obs_level_fp = np.where((y_true_reference != y_pred_reference) & (y_pred_reference == 1), 1, 0)
    obs_level_fn = np.where((y_true_reference != y_pred_reference) & (y_pred_reference == 0), 1, 0)

    combined_and_weighted = (
        obs_level_tp * business_value_matrix[1, 1]
        + obs_level_tn * business_value_matrix[0, 0]
        + obs_level_fp * business_value_matrix[0, 1]
        + obs_level_fn * business_value_matrix[1, 0]
    )

    std = np.std(combined_and_weighted)

    return (std, normalize_business_value)


def business_value_sampling_error(sampling_error_components: Tuple, data) -> float:
    """
    Calculate the false positive rate sampling error for a chunk of data.
    Parameters
    ----------
    sampling_error_components : a set of parameters that were derived from reference data.
    data : the (analysis) data you want to calculate or estimate a metric for.
    Returns
    -------
    sampling_error: float
    """
    (reference_std, norm_type) = sampling_error_components

    if norm_type is None:
        analysis_std = reference_std * len(data)
    else:  # norm_type must be 'per_prediciton'
        analysis_std = reference_std

    total_value_standard_error = analysis_std / np.sqrt(len(data))

    return total_value_standard_error
