#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import abc
import logging
from logging import Logger
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from scipy.spatial import distance

from nannyml.base import _column_is_continuous
from nannyml.chunk import Chunker
from nannyml.exceptions import InvalidArgumentsException


class Metric(abc.ABC):
    """A performance metric used to calculate realized model performance."""

    def __init__(
        self,
        display_name: str,
        column_name: str,
        calculator,
        upper_threshold_limit: float = None,
        lower_threshold_limit: float = None,
    ):
        """Creates a new Metric instance.

        Parameters
        ----------
        display_name : str
            The name of the metric. Used to display in plots. If not given this name will be derived from the
            ``calculation_function``.
        column_name: str
            The name used to indicate the metric in columns of a DataFrame.
        calculator: PerformanceCalculator
            The calculator using the Metric instance.
        upper_threshold_limit : float, default=None
            An optional upper threshold for the performance metric.
        lower_threshold_limit : float, default=None
            An optional lower threshold for the performance metric.
        """
        self.display_name = display_name
        self.column_name = column_name

        from nannyml.drift.model_inputs.univariate.distance import DistanceDriftCalculator

        if not isinstance(calculator, DistanceDriftCalculator):
            raise RuntimeError(f"{calculator.__class__.__name__} is not an instance of type " f"PerformanceCalculator")

        self.calculator: DistanceDriftCalculator = calculator

        self.upper_threshold: Optional[float] = None
        self.lower_threshold: Optional[float] = None
        self.lower_threshold_limit: Optional[float] = lower_threshold_limit
        self.upper_threshold_limit: Optional[float] = upper_threshold_limit

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

        return

    def _fit(self, reference_data: pd.DataFrame):
        raise NotImplementedError(
            f"'{self.__class__.__name__}' is a subclass of Metric and it must implement the _fit method"
        )

    def calculate(self, data: pd.DataFrame, feature_column_name: str):
        """Calculates performance metrics on data.

        Parameters
        ----------
        data: pd.DataFrame
            The data to calculate performance metrics on. Requires presence of either the predicted labels or
            prediction scores/probabilities (depending on the metric to be calculated), as well as the target data.
        """
        return self._calculate(data, feature_column_name)

    def _calculate(self, data: pd.DataFrame, feature_column_name: str):
        raise NotImplementedError(
            f"'{self.__class__.__name__}' is a subclass of Metric and it must implement the _calculate method"
        )

    def __eq__(self, other):
        """Establishes equality by comparing all properties."""
        return (
            self.display_name == other.display_name
            and self.column_name == other.column_name
            and self.upper_threshold == other.upper_threshold
            and self.lower_threshold == other.lower_threshold
        )


class MetricFactory:
    """A factory class that produces Metric instances based on a given magic string or a metric specification."""

    registry: Dict[str, Metric] = {}

    @classmethod
    def _logger(cls) -> Logger:
        return logging.getLogger(__name__)

    @classmethod
    def create(cls, key: str, kwargs: Dict[str, Any] = None) -> Metric:
        """Returns a Metric instance for a given key."""
        if not isinstance(key, str):
            raise InvalidArgumentsException(f"cannot create metric given a '{type(key)}'. Please provide a string.")

        if key not in cls.registry:
            raise InvalidArgumentsException(
                f"unknown metric key '{key}' given. " "Should be one of ['jensen_shannon']."
            )

        if kwargs is None:
            kwargs = {}

        metric_class = cls.registry[key]
        return metric_class(**kwargs)  # type: ignore

    @classmethod
    def register(cls, metric: str) -> Callable:
        def inner_wrapper(wrapped_class: Metric) -> Metric:
            if metric in cls.registry:
                cls._logger().warning(f"re-registering Metric for metric='{metric}'")
                cls.registry[metric] = wrapped_class
            else:
                cls.registry[metric] = wrapped_class
            return wrapped_class

        return inner_wrapper


@MetricFactory.register('jensen_shannon')
class JensenShannonDistance(Metric):
    """Calculates Jensen-Shannon distance."""

    def __init__(self, calculator):
        super().__init__(
            display_name='Jensen-Shannon distance',
            column_name='jensen_shannon',
            calculator=calculator,
            lower_threshold_limit=0,
        )
        self.upper_threshold = 0.1

    def _fit(self, reference_data: pd.DataFrame):
        pass

    def _calculate(self, data: pd.DataFrame, feature_column_name: str):
        ref_binned_data, ana_binned_data = get_binned_data(
            self.calculator.previous_reference_data[feature_column_name], data[feature_column_name]
        )
        return distance.jensenshannon(ref_binned_data, ana_binned_data)


def get_binned_data(reference_feature: pd.Series, analysis_feature: pd.Series):
    """Split variable into n buckets based on reference quantiles
    Args:
        reference_feature: reference data
        analysis_feature: analysis data
    Returns:
        ref_binned_pdf: probability estimate in each bucket for reference
        curr_binned_pdf: probability estimate in each bucket for reference
    """
    n_vals = reference_feature.nunique()
    if _column_is_continuous(reference_feature) and n_vals > 20:
        bins = np.histogram_bin_edges(list(reference_feature) + list(analysis_feature), bins="sturges")
        refq = pd.cut(reference_feature, bins=bins)
        anaq = pd.cut(analysis_feature, bins=bins)
        ref_binned_pdf = list(refq.value_counts(sort=False) / len(reference_feature))
        ana_binned_pdf = list(anaq.value_counts(sort=False) / len(analysis_feature))

    else:
        keys = list((set(reference_feature.unique()) | set(analysis_feature.unique())) - {np.nan})
        ref_binned_pdf = [(reference_feature == i).sum() / len(reference_feature) for i in keys]
        ana_binned_pdf = [(analysis_feature == i).sum() / len(analysis_feature) for i in keys]

    return ref_binned_pdf, ana_binned_pdf
