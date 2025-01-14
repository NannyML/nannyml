#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
"""Performance Calculation Regression Metrics Module."""

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
from nannyml.base import (
    _list_missing,
    _raise_exception_for_negative_values,
    common_nan_removal,
)
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


@MetricFactory.register(metric="mae", use_case=ProblemType.REGRESSION)
class MAE(Metric):
    """Mean Absolute Error metric."""

    y_pred: str

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        threshold: Threshold,
        y_pred_proba: Optional[str] = None,
        **kwargs,
    ):
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
            name="mae",
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            threshold=threshold,
            lower_threshold_limit=0,
            components=[("MAE", "mae")],
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def __str__(self):
        """Get string representation of metric."""
        return "MAE"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        reference_data, empty = common_nan_removal(
            reference_data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            self._sampling_error_components = (np.nan,)
        else:
            self._sampling_error_components = mae_sampling_error_components(
                y_true_reference=reference_data[self.y_true],
                y_pred_reference=reference_data[self.y_pred],
            )

    def _calculate(self, data: pd.DataFrame):
        """Redefine to handle NaNs and edge cases."""
        _list_missing([self.y_true, self.y_pred], list(data.columns))
        data, empty = common_nan_removal(
            data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            warnings.warn(
                f"No data or too many missing values, cannot calculate {self.display_name}. "
                f"Returning NaN."
            )
            return np.nan

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        return mean_absolute_error(y_true, y_pred)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        data, empty = common_nan_removal(
            data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. "
                "Returning NaN."
            )
            return np.nan
        else:
            return mae_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric="mape", use_case=ProblemType.REGRESSION)
class MAPE(Metric):
    """Mean Absolute Percentage Error metric."""

    y_pred: str

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        threshold: Threshold,
        y_pred_proba: Optional[str] = None,
        **kwargs,
    ):
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
            name="mape",
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            threshold=threshold,
            lower_threshold_limit=0,
            components=[("MAPE", "mape")],
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def __str__(self):
        """Get string representation of metric."""
        return "MAPE"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        reference_data, empty = common_nan_removal(
            reference_data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            self._sampling_error_components = (np.nan,)
        else:
            self._sampling_error_components = mape_sampling_error_components(
                y_true_reference=reference_data[self.y_true],
                y_pred_reference=reference_data[self.y_pred],
            )

    def _calculate(self, data: pd.DataFrame):
        """Redefine to handle NaNs and edge cases."""
        _list_missing([self.y_true, self.y_pred], list(data.columns))
        data, empty = common_nan_removal(
            data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            warnings.warn(
                f"No data or too many missing values, cannot calculate {self.display_name}. "
                f"Returning NaN."
            )
            return np.nan

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        return mean_absolute_percentage_error(y_true, y_pred)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        data, empty = common_nan_removal(
            data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. "
                "Returning NaN."
            )
            return np.nan
        else:
            return mape_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric="mse", use_case=ProblemType.REGRESSION)
class MSE(Metric):
    """Mean Squared Error metric."""

    y_pred: str

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        threshold: Threshold,
        y_pred_proba: Optional[str] = None,
        **kwargs,
    ):
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
            name="mse",
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            threshold=threshold,
            lower_threshold_limit=0,
            components=[("MSE", "mse")],
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def __str__(self):
        """Get string representation of metric."""
        return "MSE"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        reference_data, empty = common_nan_removal(
            reference_data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            self._sampling_error_components = (np.nan,)
        else:
            self._sampling_error_components = mse_sampling_error_components(
                y_true_reference=reference_data[self.y_true],
                y_pred_reference=reference_data[self.y_pred],
            )

    def _calculate(self, data: pd.DataFrame):
        """Redefine to handle NaNs and edge cases."""
        _list_missing([self.y_true, self.y_pred], list(data.columns))
        data, empty = common_nan_removal(
            data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            warnings.warn(
                f"No data or too many missing values, cannot calculate {self.display_name}. "
                f"Returning NaN."
            )
            return np.nan

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        return mean_squared_error(y_true, y_pred)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        data, empty = common_nan_removal(
            data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. "
                "Returning NaN."
            )
            return np.nan
        else:
            return mse_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric="msle", use_case=ProblemType.REGRESSION)
class MSLE(Metric):
    """Mean Squared Logarithmic Error metric."""

    y_pred: str

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        threshold: Threshold,
        y_pred_proba: Optional[str] = None,
        **kwargs,
    ):
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
            name="msle",
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            threshold=threshold,
            lower_threshold_limit=0,
            components=[("MSLE", "msle")],
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def __str__(self):
        """Get string representation of metric."""
        return "MSLE"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        reference_data, empty = common_nan_removal(
            reference_data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            self._sampling_error_components = (np.nan,)
        else:
            self._sampling_error_components = msle_sampling_error_components(
                y_true_reference=reference_data[self.y_true],
                y_pred_reference=reference_data[self.y_pred],
            )

    def _calculate(self, data: pd.DataFrame):
        """Redefine to handle NaNs and edge cases."""
        _list_missing([self.y_true, self.y_pred], list(data.columns))
        data, empty = common_nan_removal(
            data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            warnings.warn(
                f"No data or too many missing values, cannot calculate {self.display_name}. "
                f"Returning NaN."
            )
            return np.nan

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        # TODO: include option to drop negative values as well?
        _raise_exception_for_negative_values(y_true)
        _raise_exception_for_negative_values(y_pred)

        return mean_squared_log_error(y_true, y_pred)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        data, empty = common_nan_removal(
            data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. "
                "Returning NaN."
            )
            return np.nan
        else:
            return msle_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric="rmse", use_case=ProblemType.REGRESSION)
class RMSE(Metric):
    """Root Mean Squared Error metric."""

    y_pred: str

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        threshold: Threshold,
        y_pred_proba: Optional[str] = None,
        **kwargs,
    ):
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
            name="rmse",
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            threshold=threshold,
            lower_threshold_limit=0,
            components=[("RMSE", "rmse")],
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def __str__(self):
        """Get string representation of metric."""
        return "RMSE"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        reference_data, empty = common_nan_removal(
            reference_data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            self._sampling_error_components = (np.nan,)
        else:
            self._sampling_error_components = rmse_sampling_error_components(
                y_true_reference=reference_data[self.y_true],
                y_pred_reference=reference_data[self.y_pred],
            )

    def _calculate(self, data: pd.DataFrame):
        """Redefine to handle NaNs and edge cases."""
        _list_missing([self.y_true, self.y_pred], list(data.columns))
        data, empty = common_nan_removal(
            data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            warnings.warn(
                f"No data or too many missing values, cannot calculate {self.display_name}. "
                f"Returning NaN."
            )
            return np.nan

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        # Deal with breaking API change in sklearn 1.4
        # https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.root_mean_squared_error.html
        try:
            from sklearn.metrics import root_mean_squared_error

            return root_mean_squared_error(y_true, y_pred)
        except ImportError:
            from sklearn.metrics import mean_squared_error

            return mean_squared_error(y_true, y_pred, squared=False)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        data, empty = common_nan_removal(
            data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. "
                "Returning NaN."
            )
            return np.nan
        else:
            return rmse_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric="rmsle", use_case=ProblemType.REGRESSION)
class RMSLE(Metric):
    """Root Mean Squared Logarithmic Error metric."""

    y_pred: str

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        threshold: Threshold,
        y_pred_proba: Optional[str] = None,
        **kwargs,
    ):
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
            name="rmsle",
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            threshold=threshold,
            lower_threshold_limit=0,
            components=[("RMSLE", "rmsle")],
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def __str__(self):
        """Get string representation of metric."""
        return "RMSLE"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        reference_data, empty = common_nan_removal(
            reference_data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            self._sampling_error_components = (np.nan,)
        else:
            self._sampling_error_components = rmsle_sampling_error_components(
                y_true_reference=reference_data[self.y_true],
                y_pred_reference=reference_data[self.y_pred],
            )

    def _calculate(self, data: pd.DataFrame):
        """Redefine to handle NaNs and edge cases."""
        _list_missing([self.y_true, self.y_pred], list(data.columns))
        data, empty = common_nan_removal(
            data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            warnings.warn(
                f"No data or too many missing values, cannot calculate {self.display_name}. "
                f"Returning NaN."
            )
            return np.nan

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        # TODO: include option to drop negative values as well?
        _raise_exception_for_negative_values(y_true)
        _raise_exception_for_negative_values(y_pred)

        # Deal with breaking API change in sklearn 1.4
        # https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.root_mean_squared_log_error.html
        try:
            from sklearn.metrics import root_mean_squared_log_error

            return root_mean_squared_log_error(y_true, y_pred)
        except ImportError:
            from sklearn.metrics import mean_squared_log_error

            return mean_squared_log_error(y_true, y_pred, squared=False)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        data, empty = common_nan_removal(
            data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. "
                "Returning NaN."
            )
            return np.nan
        else:
            return rmsle_sampling_error(self._sampling_error_components, data)
