#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
)

from nannyml._typing import ProblemType
from nannyml.base import _list_missing, _raise_exception_for_negative_values, _remove_nans
from nannyml.performance_calculation.metrics.base import Metric, MetricFactory
from nannyml.sampling_error.regression import (
    mae_sampling_error,
    mae_sampling_error_components,
    mape_sampling_error,
    mape_sampling_error_components,
    mse_sampling_error,
    mse_sampling_error_components,
    msle_sampling_error,
    msle_sampling_error_components,
    rmse_sampling_error,
    rmse_sampling_error_components,
    rmsle_sampling_error,
    rmsle_sampling_error_components,
)
from nannyml.thresholds import Threshold


@MetricFactory.register(metric='mae', use_case=ProblemType.REGRESSION)
class MAE(Metric):
    """Mean Absolute Error metric."""

    def __init__(self, y_true: str, y_pred: str, threshold: Threshold, y_pred_proba: Optional[str] = None, **kwargs):
        """Creates a new MAE instance.

        Parameters
        ----------
        y_true: str
            The name of the column containing target values.
        y_pred: str
            The name of the column containing your model predictions.
        threshold: Threshold
            The Threshold instance that determines how the lower and upper threshold values will be calculated.
        y_pred_proba: Optional[str], default=None
            Name of the column containing your model output.
        """
        super().__init__(
            name='mae',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            threshold=threshold,
            lower_threshold_limit=0,
            components=[('MAE', 'mae')],
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def __str__(self):
        return "MAE"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        self._sampling_error_components = mae_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
        )

    def _calculate(self, data: pd.DataFrame):
        """Redefine to handle NaNs and edge cases."""
        _list_missing([self.y_true, self.y_pred], list(data.columns))
        data = _remove_nans(data, (self.y_true, self.y_pred))

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_true.empty:
            warnings.warn(f"'{self.y_true}' contains no data, cannot calculate {self.display_name}. Returning NaN.")
            return np.NaN
        elif y_pred.empty:
            warnings.warn(f"'{self.y_pred}' contains no data, cannot calculate {self.display_name}. Returning NaN.")
            return np.NaN

        return mean_absolute_error(y_true, y_pred)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return mae_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='mape', use_case=ProblemType.REGRESSION)
class MAPE(Metric):
    """Mean Absolute Percentage Error metric."""

    def __init__(self, y_true: str, y_pred: str, threshold: Threshold, y_pred_proba: Optional[str] = None, **kwargs):
        """Creates a new MAPE instance.

        Parameters
        ----------
        y_true: str
            The name of the column containing target values.
        y_pred: str
            The name of the column containing your model predictions.
        threshold: Threshold
            The Threshold instance that determines how the lower and upper threshold values will be calculated.
        y_pred_proba: Optional[str], default=None
            Name of the column containing your model output.
        """
        super().__init__(
            name='mape',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            threshold=threshold,
            lower_threshold_limit=0,
            components=[('MAPE', 'mape')],
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def __str__(self):
        return "MAPE"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        self._sampling_error_components = mape_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
        )

    def _calculate(self, data: pd.DataFrame):
        """Redefine to handle NaNs and edge cases."""
        _list_missing([self.y_true, self.y_pred], list(data.columns))
        data = _remove_nans(data, (self.y_true, self.y_pred))

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_true.empty:
            warnings.warn(f"'{self.y_true}' contains no data, cannot calculate {self.display_name}. Returning NaN.")
            return np.NaN
        elif y_pred.empty:
            warnings.warn(f"'{self.y_pred}' contains no data, cannot calculate {self.display_name}. Returning NaN.")
            return np.NaN

        return mean_absolute_percentage_error(y_true, y_pred)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return mape_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='mse', use_case=ProblemType.REGRESSION)
class MSE(Metric):
    """Mean Squared Error metric."""

    def __init__(self, y_true: str, y_pred: str, threshold: Threshold, y_pred_proba: Optional[str] = None, **kwargs):
        """Creates a new MSE instance.

        Parameters
        ----------
        y_true: str
            The name of the column containing target values.
        y_pred: str
            The name of the column containing your model predictions.
        threshold: Threshold
            The Threshold instance that determines how the lower and upper threshold values will be calculated.
        y_pred_proba: Optional[str], default=None
            Name of the column containing your model output.
        """
        super().__init__(
            name='mse',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            threshold=threshold,
            lower_threshold_limit=0,
            components=[('MSE', 'mse')],
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def __str__(self):
        return "MSE"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        self._sampling_error_components = mse_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
        )

    def _calculate(self, data: pd.DataFrame):
        """Redefine to handle NaNs and edge cases."""
        _list_missing([self.y_true, self.y_pred], list(data.columns))
        data = _remove_nans(data, (self.y_true, self.y_pred))

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_true.empty:
            warnings.warn(f"'{self.y_true}' contains no data, cannot calculate {self.display_name}. Returning NaN.")
            return np.NaN
        elif y_pred.empty:
            warnings.warn(f"'{self.y_pred}' contains no data, cannot calculate {self.display_name}. Returning NaN.")
            return np.NaN

        return mean_squared_error(y_true, y_pred)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return mse_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='msle', use_case=ProblemType.REGRESSION)
class MSLE(Metric):
    """Mean Squared Logarithmic Error metric."""

    def __init__(self, y_true: str, y_pred: str, threshold: Threshold, y_pred_proba: Optional[str] = None, **kwargs):
        """Creates a new MSLE instance.

        Parameters
        ----------
        y_true: str
            The name of the column containing target values.
        y_pred: str
            The name of the column containing your model predictions.
        threshold: Threshold
            The Threshold instance that determines how the lower and upper threshold values will be calculated.
        y_pred_proba: Optional[str], default=None
            Name of the column containing your model output.
        """
        super().__init__(
            name='msle',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            threshold=threshold,
            lower_threshold_limit=0,
            components=[('MSLE', 'msle')],
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def __str__(self):
        return "MSLE"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        self._sampling_error_components = msle_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
        )

    def _calculate(self, data: pd.DataFrame):
        """Redefine to handle NaNs and edge cases."""
        _list_missing([self.y_true, self.y_pred], list(data.columns))
        data = _remove_nans(data, (self.y_true, self.y_pred))

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_true.empty:
            warnings.warn(f"'{self.y_true}' contains no data, cannot calculate {self.display_name}. Returning NaN.")
            return np.NaN
        elif y_pred.empty:
            warnings.warn(f"'{self.y_pred}' contains no data, cannot calculate {self.display_name}. Returning NaN.")
            return np.NaN

        # TODO: include option to drop negative values as well?

        _raise_exception_for_negative_values(y_true)
        _raise_exception_for_negative_values(y_pred)

        return mean_squared_log_error(y_true, y_pred)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return msle_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='rmse', use_case=ProblemType.REGRESSION)
class RMSE(Metric):
    """Root Mean Squared Error metric."""

    def __init__(self, y_true: str, y_pred: str, threshold: Threshold, y_pred_proba: Optional[str] = None, **kwargs):
        """Creates a new RMSE instance.

        Parameters
        ----------
        y_true: str
            The name of the column containing target values.
        y_pred: str
            The name of the column containing your model predictions.
        threshold: Threshold
            The Threshold instance that determines how the lower and upper threshold values will be calculated.
        y_pred_proba: Optional[str], default=None
            Name of the column containing your model output.
        """
        super().__init__(
            name='rmse',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            threshold=threshold,
            lower_threshold_limit=0,
            components=[('RMSE', 'rmse')],
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def __str__(self):
        return "RMSE"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        self._sampling_error_components = rmse_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
        )

    def _calculate(self, data: pd.DataFrame):
        """Redefine to handle NaNs and edge cases."""
        _list_missing([self.y_true, self.y_pred], list(data.columns))
        data = _remove_nans(data, (self.y_true, self.y_pred))

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_true.empty:
            warnings.warn(f"'{self.y_true}' contains no data, cannot calculate {self.display_name}. Returning NaN.")
            return np.NaN
        elif y_pred.empty:
            warnings.warn(f"'{self.y_pred}' contains no data, cannot calculate {self.display_name}. Returning NaN.")
            return np.NaN

        return mean_squared_error(y_true, y_pred, squared=False)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return rmse_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='rmsle', use_case=ProblemType.REGRESSION)
class RMSLE(Metric):
    """Root Mean Squared Logarithmic Error metric."""

    def __init__(self, y_true: str, y_pred: str, threshold: Threshold, y_pred_proba: Optional[str] = None, **kwargs):
        """Creates a new RMSLE instance.

        Parameters
        ----------
        y_true: str
            The name of the column containing target values.
        y_pred: str
            The name of the column containing your model predictions.
        threshold: Threshold
            The Threshold instance that determines how the lower and upper threshold values will be calculated.
        y_pred_proba: Optional[str], default=None
            Name of the column containing your model output.
        """
        super().__init__(
            name='rmsle',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            threshold=threshold,
            lower_threshold_limit=0,
            components=[('RMSLE', 'rmsle')],
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def __str__(self):
        return "RMSLE"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        self._sampling_error_components = rmsle_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
        )

    def _calculate(self, data: pd.DataFrame):
        """Redefine to handle NaNs and edge cases."""
        _list_missing([self.y_true, self.y_pred], list(data.columns))
        data = _remove_nans(data, (self.y_true, self.y_pred))

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_true.empty:
            warnings.warn(f"'{self.y_true}' contains no data, cannot calculate {self.display_name}. Returning NaN.")
            return np.NaN
        elif y_pred.empty:
            warnings.warn(f"'{self.y_pred}' contains no data, cannot calculate {self.display_name}. Returning NaN.")
            return np.NaN

        # TODO: include option to drop negative values as well?

        _raise_exception_for_negative_values(y_true)
        _raise_exception_for_negative_values(y_pred)

        return mean_squared_log_error(y_true, y_pred, squared=False)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return rmsle_sampling_error(self._sampling_error_components, data)
