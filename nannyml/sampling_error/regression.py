#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

from typing import Tuple

import numpy as np
import pandas as pd


def mae_sampling_error_components(y_true_reference: pd.Series, y_pred_reference: pd.Series) -> Tuple:
    """
    Calculate sampling error components for Mean Absolute Error (MAE) using reference data.

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
    std = np.std(np.abs(y_true_reference - y_pred_reference))
    return (std,)


def mae_sampling_error(sampling_error_components, data) -> float:
    """
    Calculate Mean Absolute Error (MAE) sampling error for a chunk of data.

    Parameters
    ----------
    sampling_error_components : a set of parameters that were derived from reference data.
    data : the (analysis) data you want to calculate or estimate a metric for.

    Returns
    -------
    sampling_error: float

    """
    return sampling_error_components[0] / np.sqrt(len(data))


def mape_sampling_error_components(y_true_reference: pd.Series, y_pred_reference: pd.Series) -> Tuple:
    """
    Calculate sampling error components for Mean Absolute Percentage Error (MAPE) using reference data.

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
    std = np.std(np.abs(y_true_reference - y_pred_reference) / y_true_reference)
    return (std,)


def mape_sampling_error(sampling_error_components, data) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE) sampling error for a chunk of data.

    Parameters
    ----------
    sampling_error_components : a set of parameters that were derived from reference data.
    data : the (analysis) data you want to calculate or estimate a metric for.

    Returns
    -------
    sampling_error: float

    """
    return sampling_error_components[0] / np.sqrt(len(data))


def mse_sampling_error_components(y_true_reference: pd.Series, y_pred_reference: pd.Series) -> Tuple:
    """
    Calculate sampling error components for Mean Squared Error (MSE) using reference data.

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
    std = np.std((y_true_reference - y_pred_reference) ** 2)
    return (std,)


def mse_sampling_error(sampling_error_components, data) -> float:
    """
    Calculate Mean Squared Error (MSE) sampling error for a chunk of data.

    Parameters
    ----------
    sampling_error_components : a set of parameters that were derived from reference data.
    data : the (analysis) data you want to calculate or estimate a metric for.

    Returns
    -------
    sampling_error: float

    """
    return sampling_error_components[0] / np.sqrt(len(data))


def msle_sampling_error_components(y_true_reference: pd.Series, y_pred_reference: pd.Series) -> Tuple:
    """
    Calculate sampling error components for Mean Squared Logarithmic Error (MSLE) using reference data.

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
    std = np.std((np.log(1 + y_true_reference) - np.log(1 + y_pred_reference)) ** 2)
    return (std,)


def msle_sampling_error(sampling_error_components, data) -> float:
    """
    Calculate Mean Squared Logarithmic Error (MSLE) sampling error for a chunk of data.

    Parameters
    ----------
    sampling_error_components : a set of parameters that were derived from reference data.
    data : the (analysis) data you want to calculate or estimate a metric for.

    Returns
    -------
    sampling_error: float

    """
    return sampling_error_components[0] / np.sqrt(len(data))


def rmse_sampling_error_components(y_true_reference: pd.Series, y_pred_reference: pd.Series) -> Tuple:
    """
    Calculate sampling error components for Root Mean Squared Error (RMSE) using reference data.

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
    squared_error = (y_true_reference - y_pred_reference) ** 2
    squared_error_std = np.std(squared_error)
    squared_error_mean = np.mean(squared_error)
    return squared_error_std, squared_error_mean


def rmse_sampling_error(sampling_error_components, data) -> float:
    """
    Calculate Root Mean Squared Error (RMSE) sampling error for a chunk of data.

    Parameters
    ----------
    sampling_error_components : a set of parameters that were derived from reference data.
    data : the (analysis) data you want to calculate or estimate a metric for.

    Returns
    -------
    sampling_error: float

    """
    squared_error_std, squared_error_mean = sampling_error_components
    return np.sqrt((squared_error_std**2) / (4 * len(data) * squared_error_mean))


def rmsle_sampling_error_components(y_true_reference: pd.Series, y_pred_reference: pd.Series) -> Tuple:
    """
    Calculate sampling error components for Root Mean Squared Logarithmic Error (RMSLE) using reference data.

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
    squared_log_error = (np.log(1 + y_true_reference) - np.log(1 + y_pred_reference)) ** 2
    squared_log_error_std = np.std(squared_log_error)
    squared_log_error_mean = np.mean(squared_log_error)
    return squared_log_error_std, squared_log_error_mean


def rmsle_sampling_error(sampling_error_components, data) -> float:
    """
    Calculate Root Mean Squared Logarithmic Error (RMSLE) sampling error for a chunk of data.

    Parameters
    ----------
    sampling_error_components : a set of parameters that were derived from reference data.
    data : the (analysis) data you want to calculate or estimate a metric for.

    Returns
    -------
    sampling_error: float

    """
    squared_log_error_std, squared_log_error_mean = sampling_error_components
    return np.sqrt((squared_log_error_std**2) / (4 * len(data) * squared_log_error_mean))
