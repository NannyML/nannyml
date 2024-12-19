#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""This module contains the different drift detection method implementations.

The :class:`~nannyml.drift.univariate.methods.MethodFactory` will convert the drift detection method names
into an instance of the base :class:`~nannyml.drift.univariate.methods.Method` class.

The :class:`~nannyml.drift.univariate.calculator.UnivariateDriftCalculator` class will perform
the required data transformations before looping over all
:class:`~nannyml.drift.univariate.methods.Method` instances it holds and fit each on reference data
or calculate the drift value on analysis data.

"""

from __future__ import annotations

import abc
import logging
from copy import copy
from enum import Enum
from logging import Logger
from typing import Any, Callable, Dict, Optional, Type

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance

from nannyml._typing import Self
from nannyml.base import _remove_nans
from nannyml.chunk import Chunker
from nannyml.exceptions import InvalidArgumentsException, NotFittedException
from nannyml.thresholds import Threshold, calculate_threshold_values


class Method(abc.ABC):
    """A method base class to express the amount of drift between two distributions."""

    def __init__(
        self,
        display_name: str,
        column_name: str,
        chunker: Chunker,
        threshold: Threshold,
        computation_params: Optional[Dict[str, Any]] = None,
        upper_threshold_limit: Optional[float] = None,
        lower_threshold_limit: Optional[float] = None,
    ):
        """Creates a new Method instance.

        Parameters
        ----------
        display_name : str
            The name of the metric. Used to display in plots. If not given this name will be derived from the
            ``calculation_function``.
        column_name: str
            The name used to indicate the metric in columns of a DataFrame.
        chunker : Chunker
            The `Chunker` used to split the data sets into a lists of chunks.
        computation_params : dict, default=None
            A dictionary specifying parameter names and values to be used in the computation of the
            drift method.
        threshold : Threshold
            Threshold class defining threshold strategy.
        upper_threshold_limit : float, default=None
            An optional upper threshold limit for the data quality metric.
        lower_threshold_limit : float, default=0
            An optional lower threshold limit for the data quality metric.
        """
        self.display_name = display_name
        self.column_name = column_name

        self.threshold = threshold
        self.upper_threshold_value: Optional[float] = None
        self.lower_threshold_value: Optional[float] = None
        self.lower_threshold_value_limit: Optional[float] = lower_threshold_limit
        self.upper_threshold_value_limit: Optional[float] = upper_threshold_limit

        self.chunker = chunker

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def fit(self, reference_data: pd.Series, timestamps: Optional[pd.Series] = None) -> Self:
        """Fits a Method on reference data.

        Parameters
        ----------
        reference_data: pd.DataFrame
            The reference data used for fitting a Method. Must have target data available.
        timestamps: Optional[pd.Series], default=None
            A series containing the reference data Timestamps

        """
        # delegate to subclasses first, since _calculate might use properties set during fitting
        self._fit(reference_data, timestamps)

        # calculate alert thresholds by calculating the method values on reference chunks and applying the configured
        # threshold on those values. Then check with any limits
        if timestamps is not None:
            data = pd.concat([reference_data, timestamps], axis=1)
        else:
            data = reference_data.to_frame()

        reference_chunk_results = np.asarray(
            [self._calculate(chunk.data[reference_data.name]) for chunk in self.chunker.split(data)]
        )

        self.lower_threshold_value, self.upper_threshold_value = calculate_threshold_values(
            threshold=self.threshold,
            data=reference_chunk_results,
            lower_threshold_value_limit=self.lower_threshold_value_limit,
            upper_threshold_value_limit=self.upper_threshold_value_limit,
            logger=self._logger,
            metric_name=self.display_name,
        )

        return self

    def _fit(self, reference_data: pd.Series, timestamps: Optional[pd.Series] = None) -> Self:
        raise NotImplementedError(
            f"'{self.__class__.__name__}' is a subclass of Metric and it must implement the _fit method"
        )

    def calculate(self, data: pd.Series):
        """Calculates drift within data with respect to the reference data.

        Parameters
        ----------
        data: pd.DataFrame
            The data to compare to the reference data.
        """
        return self._calculate(data)

    def _calculate(self, data: pd.Series) -> float:
        raise NotImplementedError(
            f"'{self.__class__.__name__}' is a subclass of Metric and it must implement the _calculate method"
        )

    # This is currenlty required because not all Methods use the same data to evaluate alerts.
    # E.g. KS and Chi2 alerts are still based on p-values, hence each method needs to individually decide how
    # to evaluate alert conditions...
    # To be refactored / removed when custom thresholding kicks in (and p-values are no longer used)
    def alert(self, value: float):
        """Evaluates if an alert has occurred for this method on the current chunk data.

        Parameters
        ----------
        value: float
            The method value for a given chunk
        """
        return (self.lower_threshold_value is not None and value < self.lower_threshold_value) or (
            self.upper_threshold_value is not None and value > self.upper_threshold_value
        )

    def __eq__(self, other):
        """Establishes equality by comparing all properties."""
        return self.display_name == other.display_name and self.column_name == other.column_name


class FeatureType(str, Enum):
    """An enumeration indicating if a Method is applicable to continuous data, categorical data or both."""

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"


class MethodFactory:
    """A factory class that produces Method instances given a 'key' string and a 'feature_type' it supports."""

    registry: Dict[str, Dict[FeatureType, Type[Method]]] = {}

    @classmethod
    def _logger(cls) -> Logger:
        return logging.getLogger(__name__)

    @classmethod
    def create(cls, key: str, feature_type: FeatureType, **kwargs) -> Method:
        """Returns a Method instance for a given key and FeatureType.

        The value for the `key` is passed explicitly by the end user (provided within the `UnivariateDriftCalculator`
        initializer). The value for the FeatureType is provided implicitly by deducing it from the reference data upon
        fitting the `UnivariateDriftCalculator`.

        Any additional keyword arguments are passed along to the initializer of the Method.
        """
        if not isinstance(key, str):
            raise InvalidArgumentsException(f"cannot create method given a '{type(key)}'. Please provide a string.")

        if key not in cls.registry:
            raise InvalidArgumentsException(
                f"unknown method key '{key}' given. "
                "Should be one of ['kolmogorov_smirnov', 'jensen_shannon', 'wasserstein', 'chi2', "
                "'jensen_shannon', 'l_infinity', 'hellinger']."
            )

        if feature_type not in cls.registry[key]:
            raise InvalidArgumentsException(f"method {key} does not support {feature_type.value} features.")

        if kwargs is None:
            kwargs = {}

        method_class = cls.registry[key][feature_type]
        return method_class(**kwargs)

    @classmethod
    def register(cls, key: str, feature_type: FeatureType) -> Callable:
        """A decorator used to register a specific Method implementation to the factory.

        Registering a Method requires a `key` string and a FeatureType.

        The `key` sets the string value to select a Method by, e.g. `chi2` to select the Chi2-contingency implementation
        when creating a `UnivariateDriftCalculator`.

        Some Methods will only be applicable to one FeatureType,
        e.g. Kolmogorov-Smirnov can only be used with continuous
        data, Chi2-contingency only with categorical data.
        Some support multiple types however, such as the Jensen-Shannon distance.
        These can be registered multiple times, once for each FeatureType they support. The value for `key` can be
        identical, the factory will use both the FeatureType and the `key` value to determine which class
        to instantiate.

        Examples
        --------
        >>> @MethodFactory.register(key='jensen_shannon', feature_type=FeatureType.CONTINUOUS)
        >>> @MethodFactory.register(key='jensen_shannon', feature_type=FeatureType.CATEGORICAL)
        >>> class JensenShannonDistance(Method):
        ...   pass
        """

        def inner_wrapper(wrapped_class: Type[Method]) -> Type[Method]:
            if key not in cls.registry:
                cls.registry[key] = {feature_type: wrapped_class}
            else:
                if feature_type not in cls.registry[key]:
                    cls.registry[key].update({feature_type: wrapped_class})
                else:
                    cls._logger().warning(f"re-registering Method for key='{key}' and feature_type='{feature_type}'")
                    cls.registry[key][feature_type] = wrapped_class

            return wrapped_class

        return inner_wrapper


@MethodFactory.register(key='jensen_shannon', feature_type=FeatureType.CONTINUOUS)
class ContinuousJensenShannonDistance(Method):
    """Calculates Jensen-Shannon distance.

    By default an alert will be raised if `distance > 0.1`.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize Jensen-Shannon method."""
        super().__init__(
            display_name='Jensen-Shannon distance',
            column_name='jensen_shannon',
            lower_threshold_limit=0,
            **kwargs,
        )
        """
        Parameters
        ----------
        display_name : str, default='Jensen-Shannon distance'
            The name of the metric. Used to display in plots.
        column_name: str, default='jensen-shannon'
            The name used to indicate the metric in columns of a DataFrame.
        lower_threshold_limit : float, default=0
            An optional lower threshold for the performance metric.
        """
        self._bins: np.ndarray
        self._reference_proba_in_bins: np.ndarray

    def _fit(self, reference_data: pd.Series, timestamps: Optional[pd.Series] = None):
        reference_data = _remove_nans(reference_data)
        len_reference = len(reference_data)

        # Explicit conversion to float because of
        # https://github.com/numpy/numpy/commit/c63969c6e1d58e791632aacfb88ecae465d6dcfc
        bins = np.histogram_bin_edges(reference_data.astype("float64"), bins='doane')
        reference_proba_in_bins = np.histogram(reference_data, bins=bins)[0] / len_reference
        self._bins = bins
        self._reference_proba_in_bins = reference_proba_in_bins

        return self

    def _calculate(self, data: pd.Series):
        reference_proba_in_bins = copy(self._reference_proba_in_bins)
        data = _remove_nans(data)
        if data.empty:
            return np.nan

        len_data = len(data)
        data_proba_in_bins = np.histogram(data, bins=self._bins)[0] / len_data

        leftover = 1 - np.sum(data_proba_in_bins)
        if leftover > 0:
            data_proba_in_bins = np.append(data_proba_in_bins, leftover)
            reference_proba_in_bins = np.append(reference_proba_in_bins, 0)

        distance = jensenshannon(reference_proba_in_bins, data_proba_in_bins, base=2)

        return distance


@MethodFactory.register(key='jensen_shannon', feature_type=FeatureType.CATEGORICAL)
class CategoricalJensenShannonDistance(Method):
    """Calculates Jensen-Shannon distance.

    By default an alert will be raised if `distance > 0.1`.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize Jensen-Shannon method."""
        super().__init__(
            display_name='Jensen-Shannon distance',
            column_name='jensen_shannon',
            lower_threshold_limit=0,
            **kwargs,
        )
        """
        Parameters
        ----------
        display_name : str, default='Jensen-Shannon distance'
            The name of the metric. Used to display in plots.
        column_name: str, default='jensen-shannon'
            The name used to indicate the metric in columns of a DataFrame.
        lower_threshold_limit : float, default=0
            An optional lower threshold for the performance metric.
        """
        self._bins: np.ndarray
        self._reference_proba_in_bins: np.ndarray

    def _fit(self, reference_data: pd.Series, timestamps: Optional[pd.Series] = None):
        reference_data = _remove_nans(reference_data)
        reference_unique, reference_counts = np.unique(reference_data, return_counts=True)
        reference_proba_per_unique = reference_counts / len(reference_data)
        self._bins = reference_unique
        self._reference_proba_in_bins = reference_proba_per_unique

        return self

    def _calculate(self, data: pd.Series):
        reference_proba_in_bins = copy(self._reference_proba_in_bins)
        data = _remove_nans(data)
        if data.empty:
            return np.nan

        data_unique, data_counts = np.unique(data, return_counts=True)
        data_counts_dic = dict(zip(data_unique, data_counts))
        data_count_on_ref_bins = [data_counts_dic[key] if key in data_counts_dic else 0 for key in self._bins]
        data_proba_in_bins = np.array(data_count_on_ref_bins) / len(data)

        leftover = 1 - np.sum(data_proba_in_bins)
        if leftover > 0:
            data_proba_in_bins = np.append(data_proba_in_bins, leftover)
            reference_proba_in_bins = np.append(reference_proba_in_bins, 0)

        distance = jensenshannon(reference_proba_in_bins, data_proba_in_bins, base=2)

        return distance


@MethodFactory.register(key='kolmogorov_smirnov', feature_type=FeatureType.CONTINUOUS)
class KolmogorovSmirnovStatistic(Method):
    """Calculates the Kolmogorov-Smirnov d-stat.

    An alert will be raised for a Chunk if `p_value < 0.05`.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize Kolmogorov-Smirnov method."""
        super().__init__(
            display_name='Kolmogorov-Smirnov statistic',
            column_name='kolmogorov_smirnov',
            upper_threshold_limit=1,
            lower_threshold_limit=0,
            **kwargs,
        )
        """
        Parameters
        ----------
        display_name : str, default='Kolmogorov-Smirnov statistic'
            The name of the metric. Used to display in plots.
        column_name: str, default='kolmogorov-smirnov'
            The name used to indicate the metric in columns of a DataFrame.
        upper_threshold_limit : float, default=1.0
            An optional upper threshold for the performance metric.
        lower_threshold_limit : float, default=0
            An optional lower threshold for the performance metric.
        """
        self._reference_data: Optional[pd.Series] = None
        self._reference_size: float
        self._qts: np.ndarray
        self._ref_rel_freqs: Optional[np.ndarray] = None
        self._fitted = False
        if (
            (not kwargs)
            or ('computation_params' not in kwargs)
            or (self.column_name not in kwargs['computation_params'])
        ):
            self.calculation_method = 'auto'
            self.n_bins = 10_000
        else:
            self.calculation_method = kwargs['computation_params'].get('calculation_method', 'auto')
            self.n_bins = kwargs['computation_params'].get('n_bins', 10_000)

    def _fit(self, reference_data: pd.Series, timestamps: Optional[pd.Series] = None) -> Self:
        reference_data = _remove_nans(reference_data)
        if (self.calculation_method == 'auto' and len(reference_data) < 10_000) or self.calculation_method == 'exact':
            self._reference_data = reference_data
        else:
            quantile_range = np.linspace(np.min(reference_data), np.max(reference_data), self.n_bins + 1)
            # quantile_edges = np.quantile(reference_data, quantile_range)
            reference_proba_in_qts, self._qts = np.histogram(reference_data, quantile_range)
            ref_rel_freqs = reference_proba_in_qts / len(reference_data)
            self._ref_rel_freqs = np.cumsum(ref_rel_freqs)
        self._reference_size = len(reference_data)

        self._fitted = True
        return self

    def _calculate(self, data: pd.Series):
        data = _remove_nans(data)
        if data.empty:
            return np.nan
        if not self._fitted:
            raise NotFittedException(
                "tried to call 'calculate' on an unfitted method " f"{self.display_name}. Please run 'fit' first"
            )
        if (
            self.calculation_method == 'auto' and self._reference_size >= 10_000
        ) or self.calculation_method == 'estimated':
            m, n = sorted([float(self._reference_size), float(len(data))], reverse=True)
            chunk_proba_in_qts, _ = np.histogram(data, self._qts)
            chunk_rel_freqs = chunk_proba_in_qts / len(data)
            rel_freq_lower_than_edges = len(data[data < self._qts[0]]) / len(data)
            chunk_rel_freqs = rel_freq_lower_than_edges + np.cumsum(chunk_rel_freqs)
            stat = np.max(abs(self._ref_rel_freqs - chunk_rel_freqs))  # type: ignore
        else:
            stat, _ = ks_2samp(self._reference_data, data)

        return stat


@MethodFactory.register(key='chi2', feature_type=FeatureType.CATEGORICAL)
class Chi2Statistic(Method):
    """Calculates the Chi2-contingency statistic.

    An alert will be raised for a Chunk if `p_value < 0.05`.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize Chi2-contingency method."""
        super().__init__(
            display_name='Chi2 statistic',
            column_name='chi2',
            upper_threshold_limit=1.0,
            lower_threshold_limit=0,
            **kwargs,
        )
        """
        Parameters
        ----------
        display_name : str, default='Chi2 statistic'
            The name of the metric. Used to display in plots.
        column_name: str, default='chi2'
            The name used to indicate the metric in columns of a DataFrame.
        upper_threshold_limit : float, default=1.0
            An optional upper threshold for the performance metric.
        lower_threshold_limit : float, default=0
            An optional lower threshold for the performance metric.
        """
        self._reference_data_vcs: pd.Series
        self._p_value: float
        self._fitted = False

    def fit(self, reference_data: pd.Series, timestamps: Optional[pd.Series] = None) -> Self:
        """Fits Chi2 Method on reference data.

        Parameters
        ----------
        reference_data: pd.DataFrame
            The reference data used for fitting a Method. Must have target data available.
        timestamps: Optional[pd.Series], default=None
            A series containing the reference data Timestamps

        """
        super().fit(reference_data, timestamps)

        # Thresholding is based on p-values. Ignoring all custom thresholding and disable plotting a threshold
        self.lower_threshold_value = None
        self.upper_threshold_value = None
        return self

    def _fit(self, reference_data: pd.Series, timestamps: Optional[pd.Series] = None) -> Self:
        reference_data = _remove_nans(reference_data)
        self._reference_data_vcs = reference_data.value_counts().loc[lambda v: v != 0]
        self._fitted = True
        return self

    def _calculate(self, data: pd.Series):
        data = _remove_nans(data)
        if data.empty:
            return np.nan
        if not self._fitted:
            raise NotFittedException(
                "tried to call 'calculate' on an unfitted method " f"{self.display_name}. Please run 'fit' first"
            )

        stat, self._p_value = self._calc_chi2(data)
        return stat

    def alert(self, value: float):
        """Evaluates if an alert has occurred for Chi2 on the current chunk data.

        For Chi2 alerts are based on p-values rather than the actual method values like
        in all other Univariate drift methods.

        Parameters
        ----------
        value: float
            The method value for a given chunk
        """
        return self._p_value < 0.05

    def _calc_chi2(self, data: pd.Series):
        value_counts = data.value_counts().loc[lambda v: v != 0]
        stat, p_value, _, _ = chi2_contingency(
            pd.concat(
                [self._reference_data_vcs, value_counts],
                axis=1,
            ).fillna(0)
        )
        return stat, p_value


@MethodFactory.register(key='l_infinity', feature_type=FeatureType.CATEGORICAL)
class LInfinityDistance(Method):
    """Calculates the L-Infinity Distance.

    An alert will be raised if `distance > 0.1`.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize L-Infinity Distance method."""
        super().__init__(
            display_name='L-Infinity distance',
            column_name='l_infinity',
            lower_threshold_limit=0,
            **kwargs,
        )
        """
        Parameters
        ----------
        display_name : str, default='L-Infinity distance'
            The name of the metric. Used to display in plots.
        column_name: str, default='l_infinity'
            The name used to indicate the metric in columns of a DataFrame.
        lower_threshold_limit : float, default=0
            An optional lower threshold for the performance metric.
        """

        self._reference_proba: Optional[pd.Series] = None

    def _fit(self, reference_data: pd.Series, timestamps: Optional[pd.Series] = None) -> Self:
        reference_data = _remove_nans(reference_data)
        self._reference_proba = reference_data.value_counts(normalize=True)

        return self

    def _calculate(self, data: pd.Series):
        if self._reference_proba is None:
            raise NotFittedException(
                "tried to call 'calculate' on an unfitted method " f"{self.display_name}. Please run 'fit' first"
            )
        data = _remove_nans(data)
        if data.empty:
            return np.nan

        analysis_data_ratio = data.value_counts(normalize=True)
        return self._reference_proba.sub(analysis_data_ratio, fill_value=0).abs().max()


@MethodFactory.register(key='wasserstein', feature_type=FeatureType.CONTINUOUS)
class WassersteinDistance(Method):
    """Calculates the Wasserstein Distance between two distributions.

    An alert will be raised for a Chunk if .
    """

    def __init__(self, **kwargs) -> None:
        """Initialize Wasserstein Distance method."""
        super().__init__(
            display_name='Wasserstein distance',
            column_name='wasserstein',
            lower_threshold_limit=0,
            **kwargs,
        )
        """
        Parameters
        ----------
        display_name : str, default='Wasserstein distance'
            The name of the metric. Used to display in plots.
        column_name: str, default='wasserstein'
            The name used to indicate the metric in columns of a DataFrame.
        lower_threshold_limit : float, default=0
            An optional lower threshold for the performance metric.
        """

        self._reference_data: Optional[pd.Series] = None
        self._reference_size: float
        self._bin_width: float
        self._bin_edges: np.ndarray
        self._ref_rel_freqs: Optional[np.ndarray] = None
        self._ref_min: float
        self._ref_max: float
        self._ref_cdf: np.ndarray
        self._fitted = False
        if (
            (not kwargs)
            or ('computation_params' not in kwargs)
            or (self.column_name not in kwargs['computation_params'])
        ):
            self.calculation_method = 'auto'
            self.n_bins = 10_000
        else:
            self.calculation_method = kwargs['computation_params'].get('calculation_method', 'auto')
            self.n_bins = kwargs['computation_params'].get('n_bins', 10_000)

    def _fit(self, reference_data: pd.Series, timestamps: Optional[pd.Series] = None) -> Self:
        reference_data = _remove_nans(reference_data)
        if (self.calculation_method == 'auto' and len(reference_data) < 10_000) or self.calculation_method == 'exact':
            self._reference_data = reference_data
        else:
            reference_proba_in_bins, self._bin_edges = np.histogram(reference_data, bins=self.n_bins)
            self._ref_rel_freqs = reference_proba_in_bins / len(reference_data)
            self._bin_width = self._bin_edges[1] - self._bin_edges[0]
            self._ref_min = self._bin_edges[0]
            self._ref_max = self._bin_edges[-1]
            self._ref_cdf = np.cumsum(self._ref_rel_freqs)

        self._fitted = True
        self._reference_size = len(reference_data)

        return self

    def _calculate(self, data: pd.Series):
        if not self._fitted:
            raise NotFittedException(
                "tried to call 'calculate' on an unfitted method " f"{self.display_name}. Please run 'fit' first"
            )
        data = _remove_nans(data)
        if data.empty:
            return np.nan
        if (
            self.calculation_method == 'auto' and self._reference_size >= 10_000
        ) or self.calculation_method == 'estimated':
            data_smaller = data[data < self._ref_min]
            data_bigger = data[data > self._ref_max]
            n_smaller = len(data_smaller)
            n_bigger = len(data_bigger)

            if n_smaller > 0:
                amount_smaller = (n_smaller + 1) / len(data)
                smaller_with_first_ref_value = np.concatenate((data_smaller, [self._ref_min]))
                x, y = self._ecdf(smaller_with_first_ref_value)
                term_smaller = np.sum((y)[:-1] * np.diff(x))
                term_smaller = term_smaller * amount_smaller
            else:
                term_smaller, amount_smaller = 0, 0

            if n_bigger > 0:
                amount_bigger = (n_bigger + 1) / len(data)
                bigger_with_last_ref_value = np.concatenate(([self._ref_max], data_bigger))
                x, y = self._ecdf(bigger_with_last_ref_value)
                term_bigger = np.sum((1 - y)[:-1] * np.diff(x))
                term_bigger = term_bigger * amount_bigger
            else:
                term_bigger, amount_bigger = 0, 0

            data_histogram, _ = np.histogram(data, bins=self._bin_edges)
            data_histogram = data_histogram / len(data)

            data_cdf = np.cumsum(data_histogram)
            data_cdf = data_cdf + amount_smaller  # if there's some data on the left-hand side
            term_within = np.sum(np.abs(self._ref_cdf - data_cdf) * self._bin_width)

            distance = term_within + term_smaller + term_bigger
        else:
            distance = wasserstein_distance(self._reference_data, data)

        return distance

    def _ecdf(self, vec: np.ndarray):
        """Custom implementation to calculate ECDF."""
        vec = np.sort(vec)
        x, counts = np.unique(vec, return_counts=True)
        cdf = np.cumsum(counts) / len(vec)
        return x, cdf


@MethodFactory.register(key='hellinger', feature_type=FeatureType.CONTINUOUS)
class ContinuousHellingerDistance(Method):
    """Calculates the Hellinger Distance between two distributions."""

    def __init__(self, **kwargs) -> None:
        """Initialize Hellinger Distance method."""
        super().__init__(
            display_name='Hellinger distance',
            column_name='hellinger',
            lower_threshold_limit=0,
            **kwargs,
        )
        """
        Parameters
        ----------
        display_name : str, default='Hellinger distance'
            The name of the metric. Used to display in plots.
        column_name: str, default='hellinger'
            The name used to indicate the metric in columns of a DataFrame.
        lower_threshold_limit : float, default=0
            An optional lower threshold for the performance metric.
        """

        self._bins: np.ndarray
        self._reference_proba_in_bins: np.ndarray

    def _fit(self, reference_data: pd.Series, timestamps: Optional[pd.Series] = None) -> Self:
        reference_data = _remove_nans(reference_data)
        len_reference = len(reference_data)

        bins = np.histogram_bin_edges(reference_data.astype("float64"), bins='doane')
        reference_proba_in_bins = np.histogram(reference_data, bins=bins)[0] / len_reference
        self._bins = bins
        self._reference_proba_in_bins = reference_proba_in_bins

        return self

    def _calculate(self, data: pd.Series):
        data = _remove_nans(data)
        if data.empty:
            return np.nan
        reference_proba_in_bins = copy(self._reference_proba_in_bins)
        data_proba_in_bins = np.histogram(data, bins=self._bins)[0] / len(data)

        leftover = 1 - np.sum(data_proba_in_bins)
        if leftover > 0:
            data_proba_in_bins = np.append(data_proba_in_bins, leftover)
            reference_proba_in_bins = np.append(reference_proba_in_bins, 0)

        distance = np.sqrt(np.sum((np.sqrt(reference_proba_in_bins) - np.sqrt(data_proba_in_bins)) ** 2)) / np.sqrt(2)

        return distance


@MethodFactory.register(key='hellinger', feature_type=FeatureType.CATEGORICAL)
class CategoricalHellingerDistance(Method):
    """Calculates the Hellinger Distance between two distributions."""

    def __init__(self, **kwargs) -> None:
        """Initialize Hellinger Distance method."""
        super().__init__(
            display_name='Hellinger distance',
            column_name='hellinger',
            lower_threshold_limit=0,
            **kwargs,
        )
        """
        Parameters
        ----------
        display_name : str, default='Hellinger distance'
            The name of the metric. Used to display in plots.
        column_name: str, default='hellinger'
            The name used to indicate the metric in columns of a DataFrame.
        lower_threshold_limit : float, default=0
            An optional lower threshold for the performance metric.
        """

        self._bins: np.ndarray
        self._reference_proba_in_bins: np.ndarray

    def _fit(self, reference_data: pd.Series, timestamps: Optional[pd.Series] = None) -> Self:
        reference_data = _remove_nans(reference_data)

        reference_unique, reference_counts = np.unique(reference_data, return_counts=True)
        reference_proba_per_unique = reference_counts / len(reference_data)
        self._bins = reference_unique
        self._reference_proba_in_bins = reference_proba_per_unique

        return self

    def _calculate(self, data: pd.Series):
        data = _remove_nans(data)
        if data.empty:
            return np.nan
        reference_proba_in_bins = copy(self._reference_proba_in_bins)

        data_unique, data_counts = np.unique(data, return_counts=True)
        data_counts_dic = dict(zip(data_unique, data_counts))
        data_count_on_ref_bins = [data_counts_dic[key] if key in data_counts_dic else 0 for key in self._bins]
        data_proba_in_bins = np.array(data_count_on_ref_bins) / len(data)

        leftover = 1 - np.sum(data_proba_in_bins)
        if leftover > 0:
            data_proba_in_bins = np.append(data_proba_in_bins, leftover)
            reference_proba_in_bins = np.append(reference_proba_in_bins, 0)

        distance = np.sqrt(np.sum((np.sqrt(reference_proba_in_bins) - np.sqrt(data_proba_in_bins)) ** 2)) / np.sqrt(2)

        return distance
