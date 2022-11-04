#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from abc import ABC
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
)

from nannyml._typing import ProblemType
from nannyml.base import _list_missing, _raise_exception_for_negative_values
from nannyml.chunk import Chunk
from nannyml.performance_calculation.metrics.base import Metric, MetricFactory, _common_data_cleaning
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


class RegressionMetric(Metric, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(lower_threshold_limit=0, *args, **kwargs)

    def _calculate_alert_thresholds(
        self,
        reference_chunks: List[Chunk],
        std_num: int = 3,
        lower_limit: Optional[float] = None,
        upper_limit: Optional[float] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        lower_threshold, upper_threshold = super()._calculate_alert_thresholds(
            reference_chunks, std_num, lower_limit, upper_limit
        )
        if lower_threshold == 0.0:
            return None, upper_threshold
        else:
            return lower_threshold, upper_threshold


@MetricFactory.register(metric='mae', use_case=ProblemType.REGRESSION)
class MAE(RegressionMetric):
    """Mean Absolute Error metric."""

    def __init__(self, y_true: str, y_pred: str, y_pred_proba: Optional[str] = None):
        """Creates a new MAE instance."""
        super().__init__(
            display_name='MAE',
            column_name='mae',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
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

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return mean_absolute_error(y_true, y_pred)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return mae_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='mape', use_case=ProblemType.REGRESSION)
class MAPE(RegressionMetric):
    """Mean Absolute Percentage Error metric."""

    def __init__(self, y_true: str, y_pred: str, y_pred_proba: Optional[str] = None):
        """Creates a new MAPE instance."""
        super().__init__(
            display_name='MAPE',
            column_name='mape',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
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

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return mean_absolute_percentage_error(y_true, y_pred)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return mape_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='mse', use_case=ProblemType.REGRESSION)
class MSE(RegressionMetric):
    """Mean Squared Error metric."""

    def __init__(self, y_true: str, y_pred: str, y_pred_proba: Optional[str] = None):
        """Creates a new MSE instance."""
        super().__init__(
            display_name='MSE',
            column_name='mse',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
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

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return mean_squared_error(y_true, y_pred)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return mse_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='msle', use_case=ProblemType.REGRESSION)
class MSLE(RegressionMetric):
    """Mean Squared Logarithmic Error metric."""

    def __init__(self, y_true: str, y_pred: str, y_pred_proba: Optional[str] = None):
        """Creates a new MSLE instance."""
        super().__init__(
            display_name='MSLE',
            column_name='msle',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
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

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        # TODO: include option to drop negative values as well?

        _raise_exception_for_negative_values(y_true)
        _raise_exception_for_negative_values(y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return mean_squared_log_error(y_true, y_pred)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return msle_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='rmse', use_case=ProblemType.REGRESSION)
class RMSE(RegressionMetric):
    """Root Mean Squared Error metric."""

    def __init__(self, y_true: str, y_pred: str, y_pred_proba: Optional[str] = None):
        """Creates a new RMSE instance."""
        super().__init__(
            display_name='RMSE',
            column_name='rmse',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
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

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return mean_squared_error(y_true, y_pred, squared=False)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return rmse_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='rmsle', use_case=ProblemType.REGRESSION)
class RMSLE(RegressionMetric):
    """Root Mean Squared Logarithmic Error metric."""

    def __init__(self, y_true: str, y_pred: str, y_pred_proba: Optional[str] = None):
        """Creates a new RMSLE instance."""
        super().__init__(
            display_name='RMSLE',
            column_name='rmsle',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
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

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        # TODO: include option to drop negative values as well?

        _raise_exception_for_negative_values(y_true)
        _raise_exception_for_negative_values(y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return mean_squared_log_error(y_true, y_pred, squared=False)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return rmsle_sampling_error(self._sampling_error_components, data)
