#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import abc
import logging
from typing import Callable, Optional, Tuple

import numpy as np


class Threshold(abc.ABC):
    """A base class used to calculate lower and upper threshold values given one or multiple arrays.

    Any subclass should implement the abstract `thresholds` method.
    It takes an array or list of arrays and converts them into lower and upper threshold values, represented
    as a tuple of optional floats.

    A `None` threshold value is interpreted as if there is no upper or lower threshold.
    One or both values might be `None`.
    """

    @property
    def _logger(self):
        return logging.getLogger(self.__name__)

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


class ConstantThreshold(Threshold):
    """A `Thresholder` implementation that returns a constant lower and or upper threshold value.

    Attributes:
        lower: Optional[float]
            The constant lower threshold value. Defaults to `None`, meaning there is no lower threshold.
        upper: Optional[float]
            The constant upper threshold value. Defaults to `None`, meaning there is no upper threshold.

    Examples:
        >>> data = np.array(range(10))
        >>> t = ConstantThreshold(lower=None, upper=0.1)
        >>> lower, upper = t.thresholds()
        >>> print(lower, upper)
        None 0.1
    """

    def __init__(self, lower: Optional[float], upper: Optional[float]):
        """Creates a new ConstantThreshold instance.

        Args:
            lower: Optional[float]
                The constant lower threshold value. Defaults to `None`, meaning there is no lower threshold.
            upper: Optional[float]
                The constant upper threshold value. Defaults to `None`, meaning there is no upper threshold.
        """
        self.lower = lower
        self.upper = upper

    def thresholds(self, data: np.ndarray, **kwargs) -> Tuple[Optional[float], Optional[float]]:
        return self.lower, self.upper


class StandardDeviationThreshold(Threshold):
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
        >>> lower, upper = t.thresholds()
        >>> print(lower, upper)
        -4.116843969807043 13.116843969807043
    """

    def __init__(
        self,
        std_lower_multiplier: float = 3,
        std_upper_multiplier: float = 3,
        agg_func: Callable[[np.ndarray], float] = np.mean,
    ):
        """Creates a new StandardDeviationThresholder instance.

        Args:
            std_lower_multiplier: float, default=3
                The number the standard deviation of the input array will be multiplied with to form the lower offset.
                This value will be subtracted from the aggregate of the input array.
                Defaults to 3.
            std_upper_multiplier: float, default=3
                The number the standard deviation of the input array will be multiplied with to form the upper offset.
                This value will be added to the aggregate of the input array.
                Defaults to 3.
            agg_func: Callable[[np.ndarray], float], default=np.mean
                A function that will be applied to the input array to aggregate it into a single value.
                Adding the upper offset to this value will yield the upper threshold, subtracting the lower offset
                will yield the lower threshold.
        """
        self.std_lower_multiplier = std_lower_multiplier
        self.std_upper_multiplier = std_upper_multiplier
        self.agg_func = agg_func

    def thresholds(self, data: np.ndarray, **kwargs) -> Tuple[Optional[float], Optional[float]]:
        aggregate = self.agg_func(data)
        std = np.std(data)

        lower_offset = std * self.std_lower_multiplier
        upper_offset = std * self.std_upper_multiplier

        lower_threshold = aggregate - lower_offset
        upper_threshold = aggregate + upper_offset

        return lower_threshold, upper_threshold
