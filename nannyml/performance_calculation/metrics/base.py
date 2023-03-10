#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import abc
import logging
from logging import Logger
from typing import Callable, Dict, Optional, Type, Union

import numpy as np
import pandas as pd

from nannyml._typing import ProblemType
from nannyml.chunk import Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.thresholds import Threshold


class Metric(abc.ABC):
    """A performance metric used to calculate realized model performance."""

    def __init__(
        self,
        display_name: str,
        column_name: str,
        y_true: str,
        y_pred: str,
        threshold: Threshold,
        y_pred_proba: Optional[Union[str, Dict[str, str]]] = None,
        upper_threshold_limit: Optional[float] = None,
        lower_threshold_limit: Optional[float] = None,
    ):
        """Creates a new Metric instance.

        Parameters
        ----------
        display_name : str
            The name of the metric. Used to display in plots. If not given this name will be derived from the
            ``calculation_function``.
        column_name: str
            The name used to indicate the metric in columns of a DataFrame.
        upper_threshold_limit : float, default=None
            An optional upper threshold for the performance metric.
        lower_threshold_limit : float, default=None
            An optional lower threshold for the performance metric.
        """
        self.display_name = display_name
        self.column_name = column_name
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba

        self.threshold = threshold
        self.upper_threshold_value: Optional[float] = None
        self.lower_threshold_value: Optional[float] = None
        self.lower_threshold_value_limit: Optional[float] = lower_threshold_limit
        self.upper_threshold_value_limit: Optional[float] = upper_threshold_limit

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def fit(self, reference_data: pd.DataFrame, chunker: Chunker):
        """Fits a Metric on reference data.

        Parameters
        ----------
        reference_data: pd.DataFrame
            The reference data used for fitting. Must have target data available.
        chunker: Chunker
            The :class:`~nannyml.chunk.Chunker` used to split the reference data into chunks.
            This value is provided by the calling
            :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator`.

        """
        self._fit(reference_data)

        # Calculate alert thresholds
        reference_chunk_results = np.asarray([self.calculate(chunk.data) for chunk in chunker.split(reference_data)])
        self.lower_threshold_value, self.upper_threshold_value = self.threshold.thresholds(reference_chunk_results)

        # explicit None-check since value might be 0
        if self.lower_threshold_value is not None and self.lower_threshold_value_limit is not None:
            if self.lower_threshold_value < self.lower_threshold_value_limit:
                self._logger.warning(
                    f"{self.display_name} lower threshold value {self.lower_threshold_value} "
                    f"overridden by lower threshold value limit {self.lower_threshold_value_limit}"
                )
                self.lower_threshold_value = self.lower_threshold_value_limit

        # explicit None-check since value might be 0
        if self.upper_threshold_value is not None and self.upper_threshold_value_limit is not None:
            if self.upper_threshold_value > self.upper_threshold_value_limit:
                self._logger.warning(
                    f"{self.display_name} upper threshold value {self.upper_threshold_value} "
                    f"overridden by upper threshold value limit {self.upper_threshold_value_limit}"
                )
                self.upper_threshold_value = self.upper_threshold_value_limit

        return

    def _fit(self, reference_data: pd.DataFrame):
        raise NotImplementedError(
            f"'{self.__class__.__name__}' is a subclass of Metric and it must implement the _fit method"
        )

    def calculate(self, data: pd.DataFrame):
        """Calculates performance metrics on data.

        Parameters
        ----------
        data: pd.DataFrame
            The data to calculate performance metrics on. Requires presence of either the predicted labels or
            prediction scores/probabilities (depending on the metric to be calculated), as well as the target data.
        """
        return self._calculate(data)

    def _calculate(self, data: pd.DataFrame):
        raise NotImplementedError(
            f"'{self.__class__.__name__}' is a subclass of Metric and it must implement the _calculate method"
        )

    def sampling_error(self, data: pd.DataFrame):
        """Calculates the sampling error with respect to the reference data for a given chunk of data.

        Parameters
        ----------
        data: pd.DataFrame
            The data to calculate the sampling error on, with respect to the reference data.

        Returns
        -------

        sampling_error: float
            The expected sampling error.

        """
        return self._sampling_error(data)

    def _sampling_error(self, data: pd.DataFrame):
        raise NotImplementedError(
            f"'{self.__class__.__name__}' is a subclass of Metric and it must implement the _sampling_error method"
        )

    def alert(self, value: float) -> bool:
        return (self.lower_threshold_value is not None and value < self.lower_threshold_value) or (
            self.upper_threshold_value is not None and value > self.upper_threshold_value
        )

    def __eq__(self, other):
        """Establishes equality by comparing all properties."""
        return (
            self.display_name == other.display_name
            and self.column_name == other.column_name
            and self.upper_threshold_value == other.upper_threshold_value
            and self.lower_threshold_value == other.lower_threshold_value
        )


class MetricFactory:
    """A factory class that produces Metric instances based on a given magic string or a metric specification."""

    registry: Dict[str, Dict[ProblemType, Type[Metric]]] = {}

    @classmethod
    def _logger(cls) -> Logger:
        return logging.getLogger(__name__)

    @classmethod
    def create(cls, key: str, use_case: ProblemType, **kwargs) -> Metric:
        """Returns a Metric instance for a given key."""
        if not isinstance(key, str):
            raise InvalidArgumentsException(
                f"cannot create metric given a '{type(key)}'" "Please provide a string, function or Metric"
            )

        if key not in cls.registry:
            raise InvalidArgumentsException(
                f"unknown metric key '{key}' given. "
                "Should be one of ['roc_auc', 'f1', 'precision', 'recall', 'specificity', "
                "'accuracy']."
            )

        if use_case not in cls.registry[key]:
            raise RuntimeError(
                f"metric '{key}' is currently not supported for use case {use_case}. "
                "Please specify another metric or use one of these supported model types for this metric: "
                f"{[md.value for md in cls.registry[key]]}"
            )
        metric_class = cls.registry[key][use_case]
        return metric_class(**kwargs)

    @classmethod
    def register(cls, metric: str, use_case: ProblemType) -> Callable:
        def inner_wrapper(wrapped_class: Type[Metric]) -> Type[Metric]:
            if metric in cls.registry:
                if use_case in cls.registry[metric]:
                    cls._logger().warning(f"re-registering Metric for metric='{metric}' and use_case='{use_case}'")
                cls.registry[metric][use_case] = wrapped_class
            else:
                cls.registry[metric] = {use_case: wrapped_class}
            return wrapped_class

        return inner_wrapper


def _common_data_cleaning(y_true, y_pred):
    y_true, y_pred = (
        pd.Series(y_true).reset_index(drop=True),
        pd.Series(y_pred).reset_index(drop=True),
    )
    y_true = y_true[~y_pred.isna()]
    y_pred.dropna(inplace=True)

    y_pred = y_pred[~y_true.isna()]
    y_true.dropna(inplace=True)

    return y_true, y_pred
