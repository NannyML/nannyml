#  Author:   Niels Nuyttens  <niels@nannyml.com>
#            Jakub Bialek    <jabub@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing functions to estimate sampling error for multiclass classification metrics."""

from typing import List, Tuple, Union, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, average_precision_score


# How many experiments to perform when doing resampling to approximate sampling error.
N_EXPERIMENTS = 50
# Max resample size - we don't need full reference if it is too big.
MAX_RESAMPLE_SIZE = 50_000


def _standard_deviation_of_variances(components: List[Tuple], data) -> float:
    class_variances = [c[0] / (len(data) * c[1]) for c in components]
    multiclass_std = np.sqrt(np.sum(class_variances)) / len(class_variances)
    return multiclass_std


def auroc_sampling_error_components(y_true_reference: List[pd.Series], y_pred_proba_reference: List[pd.Series]):
    """Calculate sampling error components for AUROC using reference data.

    The ``y_true_reference`` and ``y_pred_proba_reference`` lists represent the binarized target values and model
    probabilities. The order of the Series in both lists should both match the list of class labels present.

    Parameters
    ----------
    y_true_reference: List[pd.Series]
        Target values for the reference dataset.
    y_pred_proba_reference: List[pd.Series]
        Prediction probability values for the reference dataset.

    Returns
    -------
    sampling_error_components: List[Tuple]
    """

    def _get_class_components(y_true, y_pred_proba):
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

        return np.var(ser_multi), fraction

    # classes = sorted(y_pred_proba_reference.keys())
    # binarized_y_true_reference_list = list(label_binarize(y_true_reference, classes=classes).T)
    # y_pred_proba_reference_list = [y_pred_proba_reference[clz] for clz in classes]

    class_components = []
    for y_true_class, y_pred_proba_class in zip(y_true_reference, y_pred_proba_reference):
        class_components.append(_get_class_components(y_true_class, y_pred_proba_class))

    return class_components


def auroc_sampling_error(sampling_error_components, data) -> float:
    """Calculate the AUROC sampling error for a chunk of data.

    Parameters
    ----------
    sampling_error_components:
        a set of parameters that were derived from reference data.
    data:
        the (analysis) data you want to calculate or estimate a metric for.

    Returns
    -------
    sampling_error: float

    """
    class_variances = [c[0] / (len(data) * c[1]) for c in sampling_error_components]
    # Experiments showed that std of class variances underestimated sampling error by 20% so we manually adjust result
    multiclass_std = np.sqrt(np.sum(class_variances)) / len(class_variances) * 1.2
    return multiclass_std


def f1_sampling_error_components(y_true_reference: List[pd.Series], y_pred_reference: List[pd.Series]):
    """Calculate sampling error components for F1 using reference data.

    The ``y_true_reference`` and ``y_pred_proba_reference`` lists represent the binarized target values and model
    probabilities. The order of the Series in both lists should both match the list of class labels present.

    Parameters
    ----------
    y_true_reference: List[pd.Series]
        Target values for the reference dataset.
    y_pred_reference: List[pd.Series]
        Prediction values for the reference dataset.

    Returns
    -------
    sampling_error_components: List[Tuple]
    """

    def _get_class_components(y_true, y_pred):
        TP = np.where((y_true == y_pred) & (y_pred == 1), 1, np.nan)
        FP = np.where((y_true != y_pred) & (y_pred == 1), 0, np.nan)
        FN = np.where((y_true != y_pred) & (y_pred == 0), 0, np.nan)

        TP = TP[~np.isnan(TP)]
        FN = FN[~np.isnan(FN)]
        FP = FP[~np.isnan(FP)]

        obs_level_f1 = np.concatenate([TP, FN, FP])
        fraction_of_relevant = len(obs_level_f1) / len(y_pred)

        return np.var(obs_level_f1), fraction_of_relevant

    class_components = []
    for y_true_class, y_pred_class in zip(y_true_reference, y_pred_reference):
        class_components.append(_get_class_components(y_true_class, y_pred_class))

    return class_components


def f1_sampling_error(sampling_error_components: List[Tuple], data) -> float:
    """Calculate the F1 sampling error for a chunk of data.

    Parameters
    ----------
    sampling_error_components:
        a set of parameters that were derived from reference data.
    data:
        the (analysis) data you want to calculate or estimate a metric for.

    Returns
    -------
    sampling_error: float

    """
    return _standard_deviation_of_variances(sampling_error_components, data)


def precision_sampling_error_components(y_true_reference: List[pd.Series], y_pred_reference: List[pd.Series]):
    """Calculate sampling error components for precision using reference data.

    The ``y_true_reference`` and ``y_pred_proba_reference`` lists represent the binarized target values and model
    probabilities. The order of the Series in both lists should both match the list of class labels present.

    Parameters
    ----------
    y_true_reference: List[pd.Series]
        Target values for the reference dataset.
    y_pred_reference: List[pd.Series]
        Prediction values for the reference dataset.

    Returns
    -------
    sampling_error_components: List[Tuple]
    """

    def _get_class_components(y_true, y_pred):
        TP = np.where((y_true == y_pred) & (y_pred == 1), 1, np.nan)
        FP = np.where((y_true != y_pred) & (y_pred == 1), 0, np.nan)

        TP = TP[~np.isnan(TP)]
        FP = FP[~np.isnan(FP)]
        obs_level_precision = np.concatenate([TP, FP])
        amount_positive_pred = np.sum(y_pred)
        fraction_of_pos_pred = amount_positive_pred / len(y_pred)

        return np.var(obs_level_precision), fraction_of_pos_pred

    class_components = []
    for y_true_class, y_pred_class in zip(y_true_reference, y_pred_reference):
        class_components.append(_get_class_components(y_true_class, y_pred_class))

    return class_components


def precision_sampling_error(sampling_error_components: List[Tuple], data) -> float:
    """Calculate the precision sampling error for a chunk of data.

    Parameters
    ----------
    sampling_error_components:
        a set of parameters that were derived from reference data.
    data:
        the (analysis) data you want to calculate or estimate a metric for.

    Returns
    -------
    sampling_error: float

    """
    return _standard_deviation_of_variances(sampling_error_components, data)


def recall_sampling_error_components(y_true_reference: List[pd.Series], y_pred_reference: List[pd.Series]):
    """Calculate sampling error components for recall using reference data.

    The ``y_true_reference`` and ``y_pred_proba_reference`` lists represent the binarized target values and model
    probabilities. The order of the Series in both lists should both match the list of class labels present.

    Parameters
    ----------
    y_true_reference: List[pd.Series]
        Target values for the reference dataset.
    y_pred_reference: List[pd.Series]
        Prediction values for the reference dataset.

    Returns
    -------
    sampling_error_components: List[Tuple]
    """

    def _get_class_components(y_true, y_pred):
        TP = np.where((y_true == y_pred) & (y_pred == 1), 1, np.nan)
        FN = np.where((y_true != y_pred) & (y_pred == 0), 0, np.nan)

        TP = TP[~np.isnan(TP)]
        FN = FN[~np.isnan(FN)]
        obs_level_recall = np.concatenate([TP, FN])
        fraction_of_relevant = sum(y_true) / len(y_pred)

        return np.var(obs_level_recall), fraction_of_relevant

    class_components = []
    for y_true_class, y_pred_class in zip(y_true_reference, y_pred_reference):
        class_components.append(_get_class_components(y_true_class, y_pred_class))

    return class_components


def recall_sampling_error(sampling_error_components: List[Tuple], data) -> float:
    """Calculate the recall sampling error for a chunk of data.

    Parameters
    ----------
    sampling_error_components:
        a set of parameters that were derived from reference data.
    data:
        the (analysis) data you want to calculate or estimate a metric for.

    Returns
    -------
    sampling_error: float

    """
    return _standard_deviation_of_variances(sampling_error_components, data)


def specificity_sampling_error_components(y_true_reference: List[pd.Series], y_pred_reference: List[pd.Series]):
    """Calculate sampling error components for specificity using reference data.

    The ``y_true_reference`` and ``y_pred_proba_reference`` lists represent the binarized target values and model
    probabilities. The order of the Series in both lists should both match the list of class labels present.

    Parameters
    ----------
    y_true_reference: List[pd.Series]
        Target values for the reference dataset.
    y_pred_reference: List[pd.Series]
        Prediction values for the reference dataset.

    Returns
    -------
    sampling_error_components: List[Tuple]
    """

    def _get_class_components(y_true, y_pred):
        TN = np.where((y_true == y_pred) & (y_pred == 0), 1, np.nan)
        FP = np.where((y_true != y_pred) & (y_pred == 1), 0, np.nan)

        TN = TN[~np.isnan(TN)]
        FP = FP[~np.isnan(FP)]
        obs_level_specificity = np.concatenate([TN, FP])
        fraction_of_relevant = len(obs_level_specificity) / len(y_pred)

        return np.var(obs_level_specificity), fraction_of_relevant

    class_components = []
    for y_true_class, y_pred_class in zip(y_true_reference, y_pred_reference):
        class_components.append(_get_class_components(y_true_class, y_pred_class))

    return class_components


def specificity_sampling_error(sampling_error_components: List[Tuple], data) -> float:
    """Calculate the specificity sampling error for a chunk of data.

    Parameters
    ----------
    sampling_error_components:
        a set of parameters that were derived from reference data.
    data:
        the (analysis) data you want to calculate or estimate a metric for.

    Returns
    -------
    sampling_error: float

    """
    return _standard_deviation_of_variances(sampling_error_components, data)


def accuracy_sampling_error_components(y_true_reference: List[pd.Series], y_pred_reference: List[pd.Series]):
    """Calculate sampling error components for accuracy using reference data.

    The ``y_true_reference`` and ``y_pred_proba_reference`` lists represent the binarized target values and model
    probabilities. The order of the Series in both lists should both match the list of class labels present.

    Parameters
    ----------
    y_true_reference: List[pd.Series]
        Target values for the reference dataset.
    y_pred_reference: List[pd.Series]
        Prediction values for the reference dataset.

    Returns
    -------
    sampling_error_components: Tuple
    """
    y_true = np.asarray(y_true_reference).astype(int)
    y_pred = np.asarray(y_pred_reference).astype(int)
    correct_table = (y_true == y_pred).all(axis=1).astype(int)

    return (np.std(correct_table),)


def accuracy_sampling_error(sampling_error_components: Tuple, data) -> float:
    """Calculate the accuracy sampling error for a chunk of data.

    Parameters
    ----------
    sampling_error_components:
        a set of parameters that were derived from reference data.
    data:
        the (analysis) data you want to calculate or estimate a metric for.

    Returns
    -------
    sampling_error: float

    """
    return sampling_error_components[0] / np.sqrt(len(data))


def multiclass_confusion_matrix_sampling_error_components(
    y_true_reference: List[pd.Series], y_pred_reference: List[pd.Series], normalize_confusion_matrix: Union[str, None]
):
    """Calculate sampling error components for CM using reference data."""
    cm = confusion_matrix(y_true_reference, y_pred_reference)

    true_marginal = cm.sum(axis=1)[:, None]
    pred_marginal = cm.sum(axis=0)[None, :]

    num_observations = len(y_true_reference)

    if normalize_confusion_matrix == 'true':
        relevant_proportions = true_marginal / num_observations
    elif normalize_confusion_matrix == 'pred':
        relevant_proportions = pred_marginal / num_observations
    elif normalize_confusion_matrix == 'all':
        relevant_proportions = 1
    else:
        relevant_proportions = None

    n_rows, n_cols = cm.shape

    stds = np.zeros((n_rows, n_cols))

    for i in range(n_rows):
        for j in range(n_cols):
            if normalize_confusion_matrix == 'true':
                obs_level_array = np.zeros(true_marginal[i, 0], dtype=int)
            elif normalize_confusion_matrix == 'pred':
                obs_level_array = np.zeros(pred_marginal[0, j], dtype=int)
            elif normalize_confusion_matrix == 'all':
                obs_level_array = np.zeros(num_observations, dtype=int)
            else:
                obs_level_array = np.zeros(num_observations, dtype=int)

            end_index = cm[i, j]
            obs_level_array[:end_index] = 1

            stds[i, j] = np.std(obs_level_array)

    return stds, relevant_proportions


def multiclass_confusion_matrix_sampling_error(sampling_error_components: Tuple, data):
    """Calculate the CM sampling error for a chunk of data."""
    reference_stds, relevant_proportions = sampling_error_components

    if relevant_proportions is None:
        standard_errors = (reference_stds / np.sqrt(len(data))) * len(data)
    else:
        standard_errors = reference_stds / np.sqrt(len(data) * relevant_proportions)

    return standard_errors


def average_precision_sampling_error_components(
    y_true_reference: List[np.ndarray], y_pred_proba_reference: List[pd.Series]
):
    """Calculate sampling error components for AP using reference data.

    The ``y_true_reference`` and ``y_pred_proba_reference`` lists represent the binarized target values and model
    probabilities. The order of the Series in both lists should both match the list of class labels present.

    Parameters
    ----------
    y_true_reference: List[np.ndarray]
        Target values for the reference dataset.
    y_pred_proba_reference: List[pd.Series]
        Prediction probability values for the reference dataset.

    Returns
    -------
    sampling_error_components: List[Tuple]
    """

    def _get_class_components(y_true_reference: np.ndarray, y_pred_proba_reference: pd.Series):
        sample_size = np.minimum(y_true_reference.shape[0] // 2, MAX_RESAMPLE_SIZE)

        y_pred_proba_reference = y_pred_proba_reference.to_numpy()

        ap_results = []
        for _ in range(N_EXPERIMENTS):
            _indexes_for_sample = np.random.choice(y_true_reference.shape[0], sample_size, replace=True)
            sample_y_true_reference = y_true_reference[_indexes_for_sample]
            sample_y_pred_proba_reference = y_pred_proba_reference[_indexes_for_sample]
            ap_results.append(average_precision_score(sample_y_true_reference, sample_y_pred_proba_reference))
        return np.var(ap_results), sample_size

    class_components = []
    for y_true_class, y_pred_proba_class in zip(y_true_reference, y_pred_proba_reference):
        class_components.append(_get_class_components(y_true_class, y_pred_proba_class))

    return class_components


def average_precision_sampling_error(sampling_error_components, data) -> float:
    """Calculate the AUROC sampling error for a chunk of data.

    Parameters
    ----------
    sampling_error_components:
        a set of parameters that were derived from reference data.
    data:
        the (chunk) data you want to calculate or estimate a metric for.

    Returns
    -------
    sampling_error: float
    """
    class_variances = [c[0] * c[1] / len(data) for c in sampling_error_components]
    multiclass_std = np.sqrt(np.mean(class_variances))
    return multiclass_std


def _calculate_business_value_per_row(
    row,
    business_value_matrix: np.ndarray,
    classes: List[str],
):
    """Helper function that calculates business value per row in a dataframe.

    Intended to be used within a pandas apply function.
    """
    cm = confusion_matrix(y_true=np.array([row.y_true]), y_pred=np.array([row.y_pred]), labels=classes)
    bv = (cm * business_value_matrix).sum()
    return bv


def business_value_sampling_error_components(
    y_true_reference: pd.Series,
    y_pred_reference: pd.Series,
    business_value_matrix: np.ndarray,
    classes: List[str],
    normalize_business_value: Optional[str],
) -> Tuple[float, Union[str, None]]:
    """Estimate sampling error for the false negative rate.

    Parameters
    ----------
    y_true_reference: pd.Series
        Target values for the reference dataset.
    y_pred_reference: pd.Series
        Predictions for the reference dataset.
    business_value_matrix: np.ndarray
        A nxn matrix of values for the business problem.
    classes: List[str]
        An alphanumerically sorted list of the unique classes in the multiclass problem
    normalize_business_value: Optional[str], default=None
        Determines how the business value will be normalized. Allowed values are None and 'per_prediction'.

    Returns
    -------
    components: tuple
    """
    data = pd.DataFrame(
        {
            'y_true': y_true_reference,
            'y_pred': y_pred_reference,
        }
    )
    bvs = data.apply(lambda x: _calculate_business_value_per_row(x, business_value_matrix, classes), axis=1)
    return (bvs.std(), normalize_business_value)


def business_value_sampling_error(sampling_error_components: Tuple, data) -> float:
    """Calculate the false positive rate sampling error for a chunk of data.

    Parameters
    ----------
    sampling_error_components:
        a set of parameters that were derived from reference data.
    data:
        the (chunk) data you want to calculate or estimate a metric for.

    Returns
    -------
    sampling_error: float
    """
    (reference_std, norm_type) = sampling_error_components
    _size = len(data)

    if norm_type is None:
        analysis_std = reference_std * _size
    else:  # norm_type must be 'per_prediciton'
        analysis_std = reference_std

    total_value_standard_error = analysis_std / np.sqrt(_size)
    return total_value_standard_error
