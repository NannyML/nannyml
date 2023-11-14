#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from __future__ import annotations

import abc
import logging
from typing import Any, Callable, ClassVar, Dict, Optional, Tuple, Type, Union

import numpy as np

from nannyml.exceptions import InvalidArgumentsException, ThresholdException


class Threshold(abc.ABC):
    """A base class used to calculate lower and upper threshold values given one or multiple arrays.

    Any subclass should implement the abstract `thresholds` method.
    It takes an array or list of arrays and converts them into lower and upper threshold values, represented
    as a tuple of optional floats.

    A `None` threshold value is interpreted as if there is no upper or lower threshold.
    One or both values might be `None`.
    """

    _registry: ClassVar[Dict[str, Type[Threshold]]] = {}

    @property
    def _logger(self):
        return logging.getLogger(self.__name__)

    def __str__(self):
        return self.__str__()

    def __repr__(self):
        return self.__class__.__name__ + str(vars(self))

    def __init_subclass__(cls, type: str, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        Threshold._registry[type] = cls

    @abc.abstractmethod
    def thresholds(self, data: np.ndarray, **kwargs) -> Tuple[Optional[float], Optional[float]]:
        """Returns lower and upper threshold values when given one or more np.ndarray instances.

        Parameters:
            data: np.ndarray
                An array of values used to calculate the thresholds on. This will most often represent a metric
                calculated on one or more sets of data, e.g. a list of F1 scores of multiple data chunks.
            kwargs: Dict[str, Any]
                Optional keyword arguments passed to the implementing subclass.

        Returns:
            lower, upper: Tuple[Optional[float], Optional[float]]
                The lower and upper threshold values. One or both might be `None`.
        """

    @classmethod
    def parse_object(cls, object: Dict[str, Any]) -> Threshold:
        """Parse object as :class:`Threshold`"""
        type = object.pop('type', '')

        try:
            threshold_cls = cls._registry[type]
        except KeyError:
            accepted_values = ', '.join(map(repr, cls._registry))
            raise InvalidArgumentsException(
                f"Expected one of {accepted_values} for threshold type, but received '{type}'"
            )

        return threshold_cls(**object)


class ConstantThreshold(Threshold, type="constant"):
    """A `Thresholder` implementation that returns a constant lower and or upper threshold value.

    Attributes:
        lower: Optional[float]
            The constant lower threshold value. Defaults to `None`, meaning there is no lower threshold.
        upper: Optional[float]
            The constant upper threshold value. Defaults to `None`, meaning there is no upper threshold.

    Raises:
        InvalidArgumentsException: raised when an argument was given using an incorrect type or name
        ThresholdException: raised when the ConstantThreshold could not be created using the given argument values

    Examples:
        >>> data = np.array(range(10))
        >>> t = ConstantThreshold(lower=None, upper=0.1)
        >>> lower, upper = t.threshold()
        >>> print(lower, upper)
        None 0.1
    """

    def __init__(self, lower: Optional[Union[float, int]] = None, upper: Optional[Union[float, int]] = None):
        """Creates a new ConstantThreshold instance.

        Args:
            lower: Optional[Union[float, int]], default=None
                The constant lower threshold value. Defaults to `None`, meaning there is no lower threshold.
            upper: Optional[Union[float, int]], default=None
                The constant upper threshold value. Defaults to `None`, meaning there is no upper threshold.

        Raises:
            InvalidArgumentsException: raised when an argument was given using an incorrect type or name
            ThresholdException: raised when the ConstantThreshold could not be created using the given argument values
        """
        self._validate_inputs(lower, upper)

        self.lower = lower
        self.upper = upper

    def thresholds(self, data: np.ndarray, **kwargs) -> Tuple[Optional[float], Optional[float]]:
        return self.lower, self.upper

    @staticmethod
    def _validate_inputs(lower: Optional[Union[float, int]] = None, upper: Optional[Union[float, int]] = None):
        if lower is not None and not isinstance(lower, (float, int)) or isinstance(lower, bool):
            raise InvalidArgumentsException(
                f"expected type of 'lower' to be 'float', 'int' or None " f"but got '{type(lower).__name__}'"
            )

        if upper is not None and not isinstance(upper, (float, int)) or isinstance(upper, bool):
            raise InvalidArgumentsException(
                f"expected type of 'upper' to be 'float', 'int' or None " f"but got '{type(upper).__name__}'"
            )

        # explicit None check is required due to special interpretation of the value 0.0 as False
        if lower is not None and upper is not None and lower >= upper:
            raise ThresholdException(f"lower threshold {lower} must be less than upper threshold {upper}")


class StandardDeviationThreshold(Threshold, type="standard_deviation"):
    """A Thresholder that offsets the mean of an array by a multiple of the standard deviation of the array values.

    This thresholder will take the aggregate of an array of values, the mean by default and add or subtract an offset
    to get the upper and lower threshold values.
    This offset is calculated as a multiplier, by default 3, times the standard deviation of the given array.

    Attributes:
        std_lower_multiplier: float
        std_upper_multiplier: float

    Examples:
        >>> data = np.array(range(10))
        >>> t = ConstantThreshold(lower=None, upper=0.1)
        >>> lower, upper = t.threshold()
        >>> print(lower, upper)
        -4.116843969807043 13.116843969807043
    """

    def __init__(
        self,
        std_lower_multiplier: Optional[Union[float, int]] = 3,
        std_upper_multiplier: Optional[Union[float, int]] = 3,
        offset_from: Callable[[np.ndarray], Any] = np.nanmean,
    ):
        """Creates a new StandardDeviationThreshold instance.

        Args:
            std_lower_multiplier: float, default=3
                The number the standard deviation of the input array will be multiplied with to form the lower offset.
                This value will be subtracted from the aggregate of the input array.
                Defaults to 3.
            std_upper_multiplier: float, default=3
                The number the standard deviation of the input array will be multiplied with to form the upper offset.
                This value will be added to the aggregate of the input array.
                Defaults to 3.
            offset_from: Callable[[np.ndarray], Any], default=np.nanmean
                A function that will be applied to the input array to aggregate it into a single value.
                Adding the upper offset to this value will yield the upper threshold, subtracting the lower offset
                will yield the lower threshold.
        """

        self._validate_inputs(std_lower_multiplier, std_upper_multiplier)

        self.std_lower_multiplier = std_lower_multiplier
        self.std_upper_multiplier = std_upper_multiplier
        self.offset_from = offset_from

    def thresholds(self, data: np.ndarray, **kwargs) -> Tuple[Optional[float], Optional[float]]:
        aggregate = self.offset_from(data)
        std = np.nanstd(data)

        lower_threshold = aggregate - std * self.std_lower_multiplier if self.std_lower_multiplier is not None else None

        upper_threshold = aggregate + std * self.std_upper_multiplier if self.std_upper_multiplier is not None else None

        return lower_threshold, upper_threshold

    @staticmethod
    def _validate_inputs(
        std_lower_multiplier: Optional[Union[float, int]] = 3, std_upper_multiplier: Optional[Union[float, int]] = 3
    ):
        if (
            std_lower_multiplier is not None
            and not isinstance(std_lower_multiplier, (float, int))
            or isinstance(std_lower_multiplier, bool)
        ):
            raise InvalidArgumentsException(
                f"expected type of 'std_lower_multiplier' to be 'float', 'int' or None "
                f"but got '{type(std_lower_multiplier).__name__}'"
            )

        if std_lower_multiplier and std_lower_multiplier < 0:
            raise ThresholdException(
                f"'std_lower_multiplier' should be greater than 0 " f"but got value {std_lower_multiplier}"
            )

        if (
            std_upper_multiplier is not None
            and not isinstance(std_upper_multiplier, (float, int))
            or isinstance(std_upper_multiplier, bool)
        ):
            raise InvalidArgumentsException(
                f"expected type of 'std_upper_multiplier' to be 'float', 'int' or None "
                f"but got '{type(std_upper_multiplier).__name__}'"
            )

        if std_upper_multiplier and std_upper_multiplier < 0:
            raise ThresholdException(
                f"'std_upper_multiplier' should be greater than 0 " f"but got value {std_upper_multiplier}"
            )


def calculate_threshold_values(
    threshold: Threshold,
    data: np.ndarray,
    lower_threshold_value_limit: Optional[float] = None,
    upper_threshold_value_limit: Optional[float] = None,
    override_using_none: bool = False,
    logger: Optional[logging.Logger] = None,
    metric_name: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Calculate lower and upper threshold values with respect to the provided Threshold and value limits.

    Parameters:
        threshold: Threshold
            The Threshold instance that determines how the lower and upper threshold values will be calculated.
        data: np.ndarray
            The data used by the Threshold instance to calculate the lower and upper threshold values.
            This will often be the values of a drift detection method or performance metric on chunks of reference data.
        lower_threshold_value_limit: Optional[float], default=None
            An optional value that serves as a limit for the lower threshold value. Any calculated lower threshold
            values that end up below this limit will be replaced by this limit value.
            The limit is often a theoretical constraint enforced by a specific drift detection method or performance
            metric.
        upper_threshold_value_limit: Optional[float], default=None
            An optional value that serves as a limit for the lower threshold value. Any calculated lower threshold
            values that end up below this limit will be replaced by this limit value.
            The limit is often a theoretical constraint enforced by a specific drift detection method or performance
            metric.
        override_using_none: bool, default=False
            When set to True use None to override threshold values that exceed value limits.
            This will prevent them from being rendered on plots.
        logger: Optional[logging.Logger], default=None
            An optional Logger instance. When provided a warning will be logged when a calculated threshold value
            gets overridden by a threshold value limit.
        metric_name: Optional[str], default=None
            When provided the metric name will be included within any log messages for additional clarity.
    """

    lower_threshold_value, upper_threshold_value = threshold.thresholds(data)

    if (
        lower_threshold_value_limit is not None
        and lower_threshold_value is not None
        and lower_threshold_value <= lower_threshold_value_limit
    ):
        override_value = None if override_using_none else lower_threshold_value_limit
        if logger:
            logger.warning(
                f"{metric_name + ' ' if metric_name else ''}lower threshold value {lower_threshold_value} "
                f"overridden by lower threshold value limit {override_value}"
            )
        lower_threshold_value = override_value

    if (
        upper_threshold_value_limit is not None
        and upper_threshold_value is not None
        and upper_threshold_value >= upper_threshold_value_limit
    ):
        override_value = None if override_using_none else upper_threshold_value_limit
        if logger:
            logger.warning(
                f"{metric_name + ' ' if metric_name else ''}upper threshold value {upper_threshold_value} "
                f"overridden by upper threshold value limit {override_value}"
            )
        upper_threshold_value = override_value

    return lower_threshold_value, upper_threshold_value
