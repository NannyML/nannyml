#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from __future__ import annotations

import abc
import logging
from copy import copy
from enum import Enum
from logging import Logger
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance

from nannyml.base import _column_is_categorical
from nannyml.chunk import Chunker
from nannyml.exceptions import InvalidArgumentsException, NotFittedException


class Method(abc.ABC):
    """A method to express the amount of drift between two distributions."""

    def __init__(
        self,
        display_name: str,
        column_name: str,
        chunker: Optional[Chunker] = None,
        upper_threshold: Optional[float] = None,
        lower_threshold: Optional[float] = None,
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

        self.upper_threshold: Optional[float] = upper_threshold
        self.lower_threshold: Optional[float] = lower_threshold
        self.lower_threshold_limit: Optional[float] = lower_threshold_limit
        self.upper_threshold_limit: Optional[float] = upper_threshold_limit

        self.chunker: Optional[Chunker] = chunker

    def fit(self, reference_data: pd.Series, timestamps: Optional[pd.Series] = None) -> Method:
        """Fits a Method on reference data.

        Parameters
        ----------
        reference_data: pd.DataFrame
            The reference data used for fitting a Method. Must have target data available.
        timestamps: Optional[pd.Series], default=None
            A series containing the reference data Timestamps

        """
        self._fit(reference_data, timestamps)

        return self

    def _fit(self, reference_data: pd.Series, timestamps: Optional[pd.Series] = None) -> Method:
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

    def _calculate(self, data: pd.Series):
        raise NotImplementedError(
            f"'{self.__class__.__name__}' is a subclass of Metric and it must implement the _calculate method"
        )

    # This is currenlty required because not all Methods use the same data to evaluate alerts.
    # E.g. KS and Chi2 alerts are still based on p-values, hence each method needs to individually decide how
    # to evaluate alert conditions...
    # To be refactored / removed when custom thresholding kicks in (and p-values are no longer used)
    def alert(self, data: pd.Series):
        """Evaluates if an alert has occurred for this method on the current chunk data.

        Parameters
        ----------
        data: pd.DataFrame
            The data to evaluate for an alert.
        """
        return self._alert(data)

    def _alert(self, data: pd.Series):
        raise NotImplementedError(
            f"'{self.__class__.__name__}' is a subclass of Metric and it must implement the _alert method"
        )

    def __eq__(self, other):
        """Establishes equality by comparing all properties."""
        return (
            self.display_name == other.display_name
            and self.column_name == other.column_name
            and self.upper_threshold == other.upper_threshold
            and self.lower_threshold == other.lower_threshold
        )


class FeatureType(str, Enum):
    """An enumeration indicating if a Method is applicable to continuous data, categorical data or both."""

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"


class MethodFactory:
    """A factory class that produces Method instances given a 'key' string and a 'feature_type' it supports."""

    registry: Dict[str, Dict[FeatureType, Method]] = {}

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
        return method_class(**kwargs)  # type: ignore

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

        def inner_wrapper(wrapped_class: Method) -> Method:
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
@MethodFactory.register(key='jensen_shannon', feature_type=FeatureType.CATEGORICAL)
class JensenShannonDistance(Method):
    """Calculates Jensen-Shannon distance.

    An alert will be raised if `distance > 0.1`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(
            display_name='Jensen-Shannon distance',
            column_name='jensen_shannon',
            lower_threshold_limit=0,
            **kwargs,
        )
        self.upper_threshold = 0.1

        self._treat_as_type: str
        self._bins: np.ndarray
        self._reference_proba_in_bins: np.ndarray

    def _fit(self, reference_data: pd.Series, timestamps: Optional[pd.Series] = None):
        if _column_is_categorical(reference_data):
            treat_as_type = 'cat'
        else:
            n_unique_values = len(np.unique(reference_data))
            len_reference = len(reference_data)
            if n_unique_values > 50 or n_unique_values / len_reference > 0.1:
                treat_as_type = 'cont'
            else:
                treat_as_type = 'cat'

        if treat_as_type == 'cont':
            bins = np.histogram_bin_edges(reference_data, bins='doane')
            reference_proba_in_bins = np.histogram(reference_data, bins=bins)[0] / len_reference
            self._bins = bins
            self._reference_proba_in_bins = reference_proba_in_bins
        else:
            reference_unique, reference_counts = np.unique(reference_data, return_counts=True)
            reference_proba_per_unique = reference_counts / len(reference_data)
            self._bins = reference_unique
            self._reference_proba_in_bins = reference_proba_per_unique

        self._treat_as_type = treat_as_type

        return self

    def _calculate(self, data: pd.Series):
        reference_proba_in_bins = copy(self._reference_proba_in_bins)
        if self._treat_as_type == 'cont':
            len_data = len(data)
            data_proba_in_bins = np.histogram(data, bins=self._bins)[0] / len_data

        else:
            data_unique, data_counts = np.unique(data, return_counts=True)
            data_counts_dic = dict(zip(data_unique, data_counts))
            data_count_on_ref_bins = [data_counts_dic[key] if key in data_counts_dic else 0 for key in self._bins]
            data_proba_in_bins = np.array(data_count_on_ref_bins) / len(data)

        leftover = 1 - np.sum(data_proba_in_bins)
        if leftover > 0:
            data_proba_in_bins = np.append(data_proba_in_bins, leftover)
            reference_proba_in_bins = np.append(reference_proba_in_bins, 0)

        distance = jensenshannon(reference_proba_in_bins, data_proba_in_bins, base=2)
        self._p_value = None

        del reference_proba_in_bins

        return distance

    def _alert(self, data: pd.Series):
        value = self.calculate(data)
        return (self.lower_threshold is not None and value < self.lower_threshold) or (
            self.upper_threshold is not None and value > self.upper_threshold
        )


@MethodFactory.register(key='kolmogorov_smirnov', feature_type=FeatureType.CONTINUOUS)
class KolmogorovSmirnovStatistic(Method):
    """Calculates the Kolmogorov-Smirnov d-stat.

    An alert will be raised for a Chunk if `p_value < 0.05`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(
            display_name='Kolmogorov-Smirnov statistic',
            column_name='kolmogorov_smirnov',
            upper_threshold_limit=1,
            lower_threshold=None,  # setting this to `None` so we don't plot the threshold (p-value based)
            **kwargs,
        )
        self._reference_data: Optional[pd.Series] = None
        self._p_value: Optional[float] = None

    def _fit(self, reference_data: pd.Series, timestamps: Optional[pd.Series] = None) -> Method:
        self._reference_data = reference_data
        return self

    def _calculate(self, data: pd.Series):
        if self._reference_data is None:
            raise NotFittedException(
                "tried to call 'calculate' on an unfitted method " f"{self.display_name}. Please run 'fit' first"
            )

        stat, p_value = ks_2samp(self._reference_data, data)
        self._p_value = p_value
        return stat

    def _alert(self, data: pd.Series):
        if self._p_value is None:
            _, self._p_value = ks_2samp(self._reference_data, data)

        alert = self._p_value < 0.05
        self._p_value = None  # just cleaning up state before running on new chunk data (optimization)

        return alert


@MethodFactory.register(key='chi2', feature_type=FeatureType.CATEGORICAL)
class Chi2Statistic(Method):
    """Calculates the Chi2-contingency statistic.

    An alert will be raised for a Chunk if `p_value < 0.05`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(
            display_name='Chi2 statistic',
            column_name='chi2',
            upper_threshold_limit=1.0,
            lower_threshold=None,  # setting this to `None` so we don't plot the threshold (p-value based)
            **kwargs,
        )
        self._reference_data: Optional[pd.Series] = None
        self._p_value: Optional[float] = None

    def _fit(self, reference_data: pd.Series, timestamps: Optional[pd.Series] = None) -> Method:
        self._reference_data = reference_data
        return self

    def _calculate(self, data: pd.Series):
        if self._reference_data is None:
            raise NotFittedException(
                "tried to call 'calculate' on an unfitted method " f"{self.display_name}. Please run 'fit' first"
            )

        stat, p_value, _, _ = chi2_contingency(
            pd.concat(
                [self._reference_data.value_counts(), data.value_counts()],  # type: ignore
                axis=1,
            ).fillna(0)
        )
        self._p_value = p_value
        return stat

    def _alert(self, data: pd.Series):
        if self._p_value is None:
            _, self._p_value, _, _ = chi2_contingency(
                pd.concat(
                    [self._reference_data.value_counts(), data.value_counts()],  # type: ignore
                    axis=1,
                ).fillna(0)
            )

        alert = self._p_value < 0.05
        self._p_value = None
        return alert


@MethodFactory.register(key='l_infinity', feature_type=FeatureType.CATEGORICAL)
class LInfinityDistance(Method):
    """Calculates the L-Infinity Distance.

    An alert will be raised if `distance > 0.1`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(
            display_name='L-Infinity distance',
            column_name='l_infinity',
            lower_threshold_limit=0,
            **kwargs,
        )

        self.upper_threshold = 0.1
        self._reference_proba: Optional[dict] = None

    def _fit(self, reference_data: pd.Series, timestamps: Optional[pd.Series] = None) -> Method:
        ref_labels = reference_data.unique()
        self._reference_proba = {label: (reference_data == label).sum() / len(reference_data) for label in ref_labels}
        return self

    def _calculate(self, data: pd.Series):
        if self._reference_proba is None:
            raise NotFittedException(
                "tried to call 'calculate' on an unfitted method " f"{self.display_name}. Please run 'fit' first"
            )

        data_labels = data.unique()
        data_ratios = {label: (data == label).sum() / len(data) for label in data_labels}

        union_labels = set(self._reference_proba.keys()) | set(data_labels)

        differences = {}
        for label in union_labels:
            differences[label] = np.abs(self._reference_proba.get(label, 0) - data_ratios.get(label, 0))

        return max(differences.values())

    def _alert(self, data: pd.Series):
        value = self._calculate(data)
        return (self.lower_threshold is not None and value < self.lower_threshold) or (
            self.upper_threshold is not None and value > self.upper_threshold
        )


@MethodFactory.register(key='wasserstein', feature_type=FeatureType.CONTINUOUS)
class WassersteinDistance(Method):
    """Calculates the Wasserstein Distance between two distributions.

    An alert will be raised for a Chunk if .
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(
            display_name='Wasserstein distance',
            column_name='wasserstein',
            **kwargs,
        )

        self._reference_data: Optional[pd.Series] = None
        self._p_value: Optional[float] = None

    def _fit(self, reference_data: pd.Series, timestamps: Optional[pd.Series] = None) -> Method:
        self._reference_data = reference_data

        if self.chunker is None:
            self.upper_threshold = 1
        else:
            # when a timestamp column is known we have to include it for chunking
            if timestamps is not None:
                data = pd.concat([reference_data, timestamps], axis=1)
            else:
                data = reference_data.to_frame()

            ref_chunk_distances = [
                self._calculate(chunk.data[reference_data.name].values.reshape(-1))
                for chunk in self.chunker.split(data)
            ]
            self.upper_threshold = np.mean(ref_chunk_distances) + 3 * np.std(ref_chunk_distances)

        self.lower_threshold = 0

        return self

    def _calculate(self, data: pd.Series):
        if self._reference_data is None:
            raise NotFittedException(
                "tried to call 'calculate' on an unfitted method " f"{self.display_name}. Please run 'fit' first"
            )

        # reshape data to be a 1d array
        # data = data.values.reshape(-1,)

        distance = wasserstein_distance(self._reference_data, data)
        self._p_value = None

        return distance

    def _alert(self, data: pd.Series):
        value = self.calculate(data)
        alert = value < self.lower_threshold or value > self.upper_threshold

        return alert


@MethodFactory.register(key='hellinger', feature_type=FeatureType.CONTINUOUS)
@MethodFactory.register(key='hellinger', feature_type=FeatureType.CATEGORICAL)
class HellingerDistance(Method):
    """Calculates the Hellinger Distance between two distributions."""

    def __init__(self, **kwargs) -> None:
        super().__init__(
            display_name='Hellinger distance',
            column_name='hellinger',
            **kwargs,
        )

        self.upper_threshold = 0.1

        self._treat_as_type: str
        self._bins: np.ndarray
        self._reference_proba_in_bins: np.ndarray

    def _fit(self, reference_data: pd.Series, timestamps: Optional[pd.Series] = None):
        if _column_is_categorical(reference_data):
            treat_as_type = 'cat'
        else:
            n_unique_values = len(np.unique(reference_data))
            len_reference = len(reference_data)
            if n_unique_values > 50 or n_unique_values / len_reference > 0.1:
                treat_as_type = 'cont'
            else:
                treat_as_type = 'cat'

        if treat_as_type == 'cont':
            bins = np.histogram_bin_edges(reference_data, bins='doane')
            reference_proba_in_bins = np.histogram(reference_data, bins=bins)[0] / len_reference
            self._bins = bins
            self._reference_proba_in_bins = reference_proba_in_bins
        else:
            reference_unique, reference_counts = np.unique(reference_data, return_counts=True)
            reference_proba_per_unique = reference_counts / len(reference_data)
            self._bins = reference_unique
            self._reference_proba_in_bins = reference_proba_per_unique

        self._treat_as_type = treat_as_type

        return self

    def _calculate(self, data: pd.Series):
        reference_proba_in_bins = copy(self._reference_proba_in_bins)
        if self._treat_as_type == 'cont':
            len_data = len(data)
            data_proba_in_bins = np.histogram(data, bins=self._bins)[0] / len_data

        else:
            data_unique, data_counts = np.unique(data, return_counts=True)
            data_counts_dic = dict(zip(data_unique, data_counts))
            data_count_on_ref_bins = [data_counts_dic[key] if key in data_counts_dic else 0 for key in self._bins]
            data_proba_in_bins = np.array(data_count_on_ref_bins) / len(data)

        leftover = 1 - np.sum(data_proba_in_bins)
        if leftover > 0:
            data_proba_in_bins = np.append(data_proba_in_bins, leftover)
            reference_proba_in_bins = np.append(reference_proba_in_bins, 0)

        distance = np.sqrt(np.sum((np.sqrt(reference_proba_in_bins) - np.sqrt(data_proba_in_bins)) ** 2)) / np.sqrt(2)
        self._p_value = None

        del reference_proba_in_bins

        return distance

    def _alert(self, data: pd.Series):
        value = self.calculate(data)
        return (self.lower_threshold is not None and value < self.lower_threshold) or (
            self.upper_threshold is not None and value > self.upper_threshold
        )
