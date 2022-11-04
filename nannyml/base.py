#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing base classes for drift calculation."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects

from nannyml.chunk import Chunker, ChunkerFactory
from nannyml.exceptions import (
    CalculatorException,
    CalculatorNotFittedException,
    EstimatorException,
    InvalidArgumentsException,
    InvalidReferenceDataException,
)


class AbstractCalculatorResult(ABC):
    """Contains the results of a calculation and provides plotting functionality.

    The result of the :meth:`~nannyml.base.AbstractCalculator.calculate` method of a
    :class:`~nannyml.base.AbstractCalculator`.

    It is an abstract class containing shared properties and methods across implementations.
    For each :class:`~nannyml.base.AbstractCalculator` class there will be a corresponding
    :class:`~nannyml.base.AbstractCalculatorResult` implementation.
    """

    DEFAULT_COLUMNS = ('key', 'chunk_index', 'start_index', 'end_index', 'start_date', 'end_date', 'period')

    def __init__(self, results_data: pd.DataFrame, *args, **kwargs):
        """Creates a new :class:`~nannyml.base.AbstractCalculatorResult` instance.

        Parameters
        ----------
        results_data: pd.DataFrame
            The data returned by the Calculator.
        """
        self.data = results_data.copy(deep=True)

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    # TODO: define more specific interface (add common arguments)
    def __len__(self):
        return len(self.data)

    @abstractmethod
    def plot(self, *args, **kwargs) -> Optional[plotly.graph_objects.Figure]:
        """Plots calculation results."""
        raise NotImplementedError

    def to_df(self, multilevel: bool = True) -> pd.DataFrame:
        if multilevel:
            return self.data
        else:
            column_names = [
                '_'.join(col).replace('chunk_chunk_chunk', 'chunk').replace('chunk_chunk', 'chunk')
                for col in self.data.columns.values
            ]
            single_level_data = self.data.copy(deep=True)
            single_level_data.columns = column_names
            return single_level_data

    def filter(self, period: str = 'analysis', metrics: List[str] = None, *args, **kwargs) -> AbstractCalculatorResult:
        """Returns result metric data."""
        try:
            return self._filter(period, metrics, *args, **kwargs)
        except Exception as exc:
            raise CalculatorException(f"could not read result data: {exc}")

    @abstractmethod
    def _filter(self, period: str, metrics: List[str] = None, *args, **kwargs) -> AbstractCalculatorResult:
        raise NotImplementedError


class AbstractCalculator(ABC):
    """Base class for drift calculation."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_number: int = None,
        chunk_period: str = None,
        chunker: Chunker = None,
        timestamp_column_name: Optional[str] = None,
    ):
        """Creates a new instance of an abstract DriftCalculator.

        Parameters
        ----------
        chunk_size: int
            Splits the data into chunks containing `chunks_size` observations.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunk_number: int
            Splits the data into `chunk_number` pieces.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunk_period: str
            Splits the data according to the given period.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunker : Chunker
            The `Chunker` used to split the data sets into a lists of chunks.
        """
        self.chunker = ChunkerFactory.get_chunker(
            chunk_size, chunk_number, chunk_period, chunker, timestamp_column_name
        )

        self.timestamp_column_name = timestamp_column_name

        self.result: Optional[AbstractCalculatorResult] = None

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> AbstractCalculator:
        """Trains the calculator using reference data."""
        try:
            self._logger.debug(f"fitting {str(self)}")
            return self._fit(reference_data, *args, **kwargs)
        except InvalidArgumentsException:
            raise
        except InvalidReferenceDataException:
            raise
        except Exception as exc:
            raise CalculatorException(f"failed while fitting {str(self)}.\n{exc}")

    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> Any:
        """Performs a calculation on the provided data."""
        try:
            self._logger.debug(f"calculating {str(self)}")
            data = data.copy()
            return self._calculate(data, *args, **kwargs)
        except InvalidArgumentsException:
            raise
        except CalculatorNotFittedException:
            raise
        except Exception as exc:
            raise CalculatorException(f"failed while calculating {str(self)}.\n{exc}")

    @abstractmethod
    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> AbstractCalculator:
        raise NotImplementedError(f"'{self.__class__.__name__}' must implement the '_fit' method")

    @abstractmethod
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> Any:
        raise NotImplementedError(f"'{self.__class__.__name__}' must implement the '_calculate' method")


class AbstractEstimatorResult(ABC):
    """Contains the results of a drift calculation and provides additional functionality such as plotting.

    The result of the :meth:`~nannyml.drift.base.DriftCalculator.calculate` method of a
    :class:`~nannyml.drift.base.DriftCalculator`.

    It is an abstract class containing shared properties and methods across implementations.
    For each :class:`~nannyml.drift.base.DriftCalculator` class there will be an associated
    :class:`~nannyml.drift.base.DriftResult` implementation.
    """

    DEFAULT_COLUMNS = ['key', 'chunk_index', 'start_index', 'end_index', 'start_date', 'end_date', 'period']

    def __init__(self, results_data: pd.DataFrame):
        """Creates a new DriftResult instance.

        Parameters
        ----------
        results_data: pd.DataFrame
            The result data of the performed calculation.
        """
        self.data = results_data.copy(deep=True)

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def to_df(self, multilevel: bool = True):
        if multilevel:
            return self.data
        else:
            column_names = [
                '_'.join(col).replace('chunk_chunk_chunk', 'chunk').replace('chunk_chunk', 'chunk')
                for col in self.data.columns.values
            ]
            single_level_data = self.data.copy(deep=True)
            single_level_data.columns = column_names
            return single_level_data

    def filter(self, period: str = 'analysis', metrics: List[str] = None, *args, **kwargs) -> AbstractEstimatorResult:
        """Returns result metric data."""
        try:
            return self._filter(period, metrics, *args, **kwargs)
        except Exception as exc:
            raise EstimatorException(f"could not read result data: {exc}")

    @abstractmethod
    def _filter(self, period: str, metrics: List[str] = None, *args, **kwargs) -> AbstractEstimatorResult:
        raise NotImplementedError

    def plot(self, *args, **kwargs) -> plotly.graph_objects.Figure:
        """Plot drift results."""
        raise NotImplementedError


class AbstractEstimator(ABC):
    """Base class for drift calculation."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_number: int = None,
        chunk_period: str = None,
        chunker: Chunker = None,
        timestamp_column_name: str = None,
    ):
        """Creates a new instance of an abstract DriftCalculator.

        Parameters
        ----------
        chunk_size: int
            Splits the data into chunks containing `chunks_size` observations.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunk_number: int
            Splits the data into `chunk_number` pieces.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunk_period: str
            Splits the data according to the given period.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunker : Chunker
            The `Chunker` used to split the data sets into a lists of chunks.
        """
        self.chunker = ChunkerFactory.get_chunker(
            chunk_size, chunk_number, chunk_period, chunker, timestamp_column_name
        )
        self.timestamp_column_name = timestamp_column_name

        self.result: Optional[AbstractEstimatorResult] = None

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __str__(self):
        return self.__class__.__name__

    def fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> AbstractEstimator:
        """Trains the calculator using reference data."""
        try:
            self._logger.info(f"fitting {str(self)}")
            reference_data = reference_data.copy()
            return self._fit(reference_data, *args, **kwargs)
        except InvalidArgumentsException:
            raise
        except InvalidReferenceDataException:
            raise
        except Exception as exc:
            raise CalculatorException(f"failed while fitting {str(self)}.\n{exc}")

    def estimate(self, data: pd.DataFrame, *args, **kwargs) -> Any:
        """Performs a calculation on the provided data."""
        try:
            self._logger.info(f"estimating {str(self)}")
            data = data.copy()
            return self._estimate(data, *args, **kwargs)
        except InvalidArgumentsException:
            raise
        except CalculatorNotFittedException:
            raise
        except Exception as exc:
            raise CalculatorException(f"failed while calculating {str(self)}.\n{exc}")

    @abstractmethod
    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> AbstractEstimator:
        raise NotImplementedError(f"'{self.__class__.__name__}' must implement the '_fit' method")

    @abstractmethod
    def _estimate(self, data: pd.DataFrame, *args, **kwargs) -> Any:
        raise NotImplementedError(f"'{self.__class__.__name__}' must implement the '_calculate' method")


def _split_features_by_type(data: pd.DataFrame, feature_column_names: List[str]) -> Tuple[List[str], List[str]]:
    continuous_column_names = [col for col in feature_column_names if _column_is_continuous(data[col])]

    categorical_column_names = [col for col in feature_column_names if _column_is_categorical(data[col])]

    return continuous_column_names, categorical_column_names


def _column_is_categorical(column: pd.Series) -> bool:
    return column.dtype in ['object', 'string', 'category', 'bool']


def _column_is_continuous(column: pd.Series) -> bool:
    return column.dtype in [
        'int_',
        'int8',
        'int16',
        'int32',
        'int64',
        'uint8',
        'uint16',
        'uint32',
        'uint64',
        'float_',
        'float16',
        'float32',
        'float64',
    ]


def _list_missing(columns_to_find: List, dataset_columns: Union[List, pd.DataFrame]):
    if isinstance(dataset_columns, pd.DataFrame):
        dataset_columns = dataset_columns.columns

    missing = [col for col in columns_to_find if col not in dataset_columns]
    if len(missing) > 0:
        raise InvalidArgumentsException(f"missing required columns '{missing}' in data set:\n\t{dataset_columns}")


def _raise_exception_for_negative_values(column: pd.Series):
    """Raises an InvalidArgumentsException if a given column contains negative values.

    Parameters
    ----------
    column: pd.Series
        Column to check for negative values.

    Raises
    ------
    nannyml.exceptions.InvalidArgumentsException
    """
    if any(column.values < 0):
        negative_item_indices = np.where(column.values < 0)
        raise InvalidArgumentsException(
            f"target values '{column.name}' contain negative values.\n"
            f"\tLog-based metrics are not supported for negative target values.\n"
            f"\tCheck '{column.name}' at rows {str(negative_item_indices)}."
        )
