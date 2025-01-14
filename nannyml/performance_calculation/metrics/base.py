#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
"""Base Classes for performane calculation."""
import abc
import logging
from logging import Logger
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

from nannyml._typing import ProblemType
from nannyml.chunk import Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.thresholds import Threshold, calculate_threshold_values


class Metric(abc.ABC):
    """A performance metric used to calculate realized model performance."""

    def __init__(
        self,
        name: str,
        y_true: str,
        components: List[Tuple[str, str]],
        threshold: Threshold,
        y_pred: Optional[str] = None,
        y_pred_proba: Optional[Union[str, Dict[str, str]]] = None,
        upper_threshold_limit: Optional[float] = None,
        lower_threshold_limit: Optional[float] = None,
        **kwargs,
    ):
        """Creates a new Metric instance.

        Parameters
        ----------
        name: str
            The name used to indicate the metric in columns of a DataFrame.
        y_true: str
            The name of the column containing target values.
        y_pred: str
            The name of the column containing your model predictions.
        components: List[Tuple[str, str]]
            A list of (display_name, column_name) tuples. The
            display_name is used for display purposes, while the
            column_name is used for column names in the output.
        threshold: Threshold
            The Threshold instance that determines how the lower and upper threshold values will be calculated.
        y_pred_proba: Optional[Union[str, Dict[str, str]]], default=None
            Name(s) of the column(s) containing your model output.
            - For binary classification, pass a single string refering to the model output column.
            - For multiclass classification, pass a dictionary that maps a class string to the column name \
                containing model outputs for that class.
        upper_threshold_limit : float, default=None
            An optional upper threshold for the performance metric.
        lower_threshold_limit : float, default=None
            An optional lower threshold for the performance metric.
        """
        self.name: str = name

        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba

        self.threshold = threshold
        self.upper_threshold_value: Optional[float] = None
        self.lower_threshold_value: Optional[float] = None
        self.lower_threshold_value_limit: Optional[float] = lower_threshold_limit
        self.upper_threshold_value_limit: Optional[float] = upper_threshold_limit

        # A list of (display_name, column_name) tuples
        self.components: List[Tuple[str, str]] = components

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
        self.lower_threshold_value, self.upper_threshold_value = calculate_threshold_values(
            threshold=self.threshold,
            data=reference_chunk_results,
            lower_threshold_value_limit=self.lower_threshold_value_limit,
            upper_threshold_value_limit=self.upper_threshold_value_limit,
            logger=self._logger,
            metric_name=self.display_name,
        )

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
        """Returns True if a calculated metric value is below a lower threshold or above an upper threshold.

        Parameters
        ----------
        value: float
            Value of a calculated metric.

        Returns
        -------
        bool: bool
        """
        return (self.lower_threshold_value is not None and value < self.lower_threshold_value) or (
            self.upper_threshold_value is not None and value > self.upper_threshold_value
        )

    def __eq__(self, other):
        """Establishes equality by comparing all properties."""
        return (
            self.display_name == other.display_name
            and self.column_name == other.column_name
            and self.components == other.components
            and self.upper_threshold_value == other.upper_threshold_value
            and self.lower_threshold_value == other.lower_threshold_value
        )

    def get_chunk_record(self, chunk_data: pd.DataFrame) -> Dict:
        """Returns a DataFrame containing the performance metrics for a given chunk."""
        if len(self.components) > 1:
            raise NotImplementedError(
                "cannot use default 'get_chunk_record' implementation when a metric has multiple components."
            )

        column_name = self.components[0][1]

        chunk_record = {}

        try:
            realized_value = self.calculate(chunk_data)
            sampling_error = self.sampling_error(chunk_data)

            chunk_record[f'{column_name}_sampling_error'] = sampling_error
            chunk_record[f'{column_name}'] = realized_value
            chunk_record[f'{column_name}_upper_threshold'] = self.upper_threshold_value
            chunk_record[f'{column_name}_lower_threshold'] = self.lower_threshold_value
            chunk_record[f'{column_name}_alert'] = self.alert(realized_value)
        except Exception as exc:
            if self._logger:
                self._logger.error(
                    f"an unexpected exception occurred during calculation of method '{self.display_name}': " f"{exc}"
                )
            chunk_record[f'{column_name}_sampling_error'] = np.nan
            chunk_record[f'{column_name}'] = np.nan
            chunk_record[f'{column_name}_upper_threshold'] = self.upper_threshold_value
            chunk_record[f'{column_name}_lower_threshold'] = self.lower_threshold_value
            chunk_record[f'{column_name}_alert'] = np.nan
        finally:
            return chunk_record

    @property
    def display_name(self) -> str:
        """Get metric display name."""
        return self.name

    @property
    def column_name(self) -> str:
        """Get metric column name."""
        return self.components[0][1]

    @property
    def display_names(self) -> List[str]:
        """Get metric display names."""
        return [c[0] for c in self.components]

    @property
    def column_names(self) -> List[str]:
        """Get metric column names."""
        return [c[1] for c in self.components]


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
                "'accuracy', 'confusion_matrix', 'business_value']."
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
        """Register performance metric class in MetricFactory."""

        def inner_wrapper(wrapped_class: Type[Metric]) -> Type[Metric]:
            if metric in cls.registry:
                if use_case in cls.registry[metric]:
                    cls._logger().warning(f"re-registering Metric for metric='{metric}' and use_case='{use_case}'")
                cls.registry[metric][use_case] = wrapped_class
            else:
                cls.registry[metric] = {use_case: wrapped_class}
            return wrapped_class

        return inner_wrapper
