#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing base classes for drift calculation."""
from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from typing import Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union, overload

import numpy as np
import pandas as pd
import plotly.graph_objects

from nannyml._typing import Key, Metric, Result, Self
from nannyml.chunk import Chunker, ChunkerFactory
from nannyml.exceptions import CalculatorException, EstimatorException, InvalidArgumentsException, NannyMLException

MetricLike = TypeVar('MetricLike', bound=Metric)


class AbstractResult(ABC):
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

    @property
    def empty(self) -> bool:
        return self.data is None or self.data.empty

    # TODO: define more specific interface (add common arguments)
    def __len__(self):  # noqa: D105
        return len(self.data)

    @abstractmethod
    def plot(self, *args, **kwargs) -> plotly.graph_objects.Figure:
        """Plots calculation results."""
        raise NotImplementedError

    def to_df(self, multilevel: bool = True) -> pd.DataFrame:
        """Export results to pandas dataframe."""
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

    def filter(self, period: str = 'all', metrics: Optional[Union[str, List[str]]] = None, *args, **kwargs) -> Self:
        """Returns filtered result metric data."""
        if metrics and not isinstance(metrics, (str, list)):
            raise InvalidArgumentsException("metrics value provided is not a valid metric or list of metrics")
        if isinstance(metrics, str):
            metrics = [metrics]
        try:
            return self._filter(period, metrics, *args, **kwargs)
        except NannyMLException:
            raise
        except Exception as exc:
            raise CalculatorException(f"could not read result data: {exc}")

    @abstractmethod
    def _filter(self, period: str, metrics: Optional[List[str]] = None, *args, **kwargs) -> Self:
        raise NotImplementedError(f"'{self.__class__.__name__}' must implement the '_filter' method")

    @abstractmethod
    def keys(self) -> List[Key]:
        raise NotImplementedError(f"'{self.__class__.__name__}' must implement the 'items' method")

    def values(self, key: Key) -> Optional[pd.Series]:
        return self._get_property_for_key(key, property_name='value')

    def alerts(self, key: Key) -> Optional[pd.Series]:
        return self._get_property_for_key(key, property_name='alert')

    def upper_thresholds(self, key: Key) -> Optional[pd.Series]:
        return self._get_property_for_key(key, property_name='upper_threshold')

    def lower_thresholds(self, key: Key) -> Optional[pd.Series]:
        return self._get_property_for_key(key, property_name='lower_threshold')

    def upper_confidence_bounds(self, key: Key) -> Optional[pd.Series]:
        return self._get_property_for_key(key, property_name='upper_confidence_boundary')

    def lower_confidence_bounds(self, key: Key) -> Optional[pd.Series]:
        return self._get_property_for_key(key, property_name='lower_confidence_boundary')

    def sampling_error(self, key: Key) -> Optional[pd.Series]:
        return self._get_property_for_key(key, property_name='sampling_error')

    def _get_property_for_key(self, key: Key, property_name: str) -> Optional[pd.Series]:
        return self.data.get(key.properties + (property_name,), default=None)


class Abstract1DResult(AbstractResult, ABC):
    def __init__(self, results_data: pd.DataFrame, *args, **kwargs):
        super().__init__(results_data)

    @property
    def chunk_keys(self) -> pd.Series:
        return self.data[('chunk', 'key')]

    @property
    def chunk_start_dates(self) -> pd.Series:
        return self.data[('chunk', 'start_date')]

    @property
    def chunk_end_dates(self) -> pd.Series:
        return self.data[('chunk', 'end_date')]

    @property
    def chunk_start_indices(self) -> pd.Series:
        return self.data[('chunk', 'start_index')]

    @property
    def chunk_end_indices(self) -> pd.Series:
        return self.data[('chunk', 'end_index')]

    @property
    def chunk_indices(self) -> pd.Series:
        return self.data[('chunk', 'chunk_index')]

    @property
    def chunk_periods(self) -> pd.Series:
        return self.data[('chunk', 'period')]

    @property
    def chunk_start_index(self) -> pd.Series:
        return self.data[('chunk', 'start_index')]

    def _filter(self, period: str, *args, **kwargs) -> Self:
        data = self.data
        if period != 'all':
            data = self.data.loc[self.data.loc[:, ('chunk', 'period')] == period, :]
            data = data.reset_index(drop=True)

        res = copy.deepcopy(self)
        res.data = data
        return res


class PerMetricResult(Abstract1DResult, ABC, Generic[MetricLike]):
    def __init__(self, results_data: pd.DataFrame, metrics: list[MetricLike] = [], *args, **kwargs):
        super().__init__(results_data)
        self.metrics = metrics

    def _filter(self, period: str, metrics: Optional[List[str]] = None, *args, **kwargs) -> Self:
        if metrics is None:
            metrics = [metric.column_name for metric in self.metrics]

        res = super()._filter(period, *args, **kwargs)

        data = pd.concat([res.data.loc[:, (['chunk'])], res.data.loc[:, (metrics,)]], axis=1)
        data = data.reset_index(drop=True)

        res.data = data
        res.metrics = [metric for metric in self.metrics if metric.column_name in metrics]

        return res


class PerColumnResult(Abstract1DResult, ABC):
    def __init__(self, results_data: pd.DataFrame, column_names: Union[str, List[str]] = [], *args, **kwargs):
        super().__init__(results_data)
        if isinstance(column_names, str):
            self.column_names = [column_names]
        elif isinstance(column_names, list):
            self.column_names = column_names
        else:
            raise TypeError("column_names should be either a column name string or a list of strings.")

    def _filter(
        self,
        period: str,
        metrics: Optional[List[str]] = None,
        column_names: Optional[Union[str, List[str]]] = None,
        *args,
        **kwargs,
    ) -> Self:
        if isinstance(column_names, str):
            column_names = [column_names]
        elif isinstance(column_names, list):
            pass
        elif column_names is None:
            column_names = self.column_names
        else:
            raise TypeError("column_names should be either a column name string or a list of strings.")

        res = super()._filter(period, *args, **kwargs)

        data = pd.concat([res.data.loc[:, (['chunk'])], res.data.loc[:, (column_names,)]], axis=1)
        data = data.reset_index(drop=True)

        res.data = data
        res.column_names = [c for c in self.column_names if c in column_names]
        return res


class Abstract2DResult(AbstractResult, ABC):
    def __init__(self, results_data: pd.DataFrame, *args, **kwargs):
        super().__init__(results_data)

    @property
    def chunk_keys(self) -> pd.Series:
        return self.data[('chunk', 'chunk', 'key')]

    @property
    def chunk_start_dates(self) -> pd.Series:
        return self.data[('chunk', 'chunk', 'start_date')]

    @property
    def chunk_end_dates(self) -> pd.Series:
        return self.data[('chunk', 'chunk', 'end_date')]

    @property
    def chunk_start_indices(self) -> pd.Series:
        return self.data[('chunk', 'chunk', 'start_index')]

    @property
    def chunk_end_indices(self) -> pd.Series:
        return self.data[('chunk', 'chunk', 'end_index')]

    @property
    def chunk_indices(self) -> pd.Series:
        return self.data[('chunk', 'chunk', 'chunk_index')]

    @property
    def chunk_periods(self) -> pd.Series:
        return self.data[('chunk', 'chunk', 'period')]

    @property
    def chunk_start_index(self) -> pd.Series:
        return self.data[('chunk', 'chunk', 'start_index')]

    def _filter(
        self,
        period: str,
        *args,
        **kwargs,
    ) -> Self:
        data = self.data
        if period != 'all':
            data = data.loc[self.data.loc[:, ('chunk', 'chunk', 'period')] == period, :]
            data = data.reset_index(drop=True)

        res = copy.deepcopy(self)
        res.data = data

        return res


class PerMetricPerColumnResult(Abstract2DResult, ABC, Generic[MetricLike]):
    def __init__(
        self, results_data: pd.DataFrame, metrics: list[MetricLike] = [], column_names: List[str] = [], *args, **kwargs
    ):
        super().__init__(results_data)
        self.metrics = metrics
        self.column_names = column_names

    def _filter(
        self,
        period: str,
        metrics: Optional[List[str]] = None,
        column_names: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> Self:
        res = super()._filter(period, *args, **kwargs)
        if metrics is None and column_names is None:
            return res

        if metrics is None:
            metrics = [metric.column_name for metric in self.metrics]
        if column_names is None:
            column_names = self.column_names

        data = pd.concat([res.data.loc[:, (['chunk'])], res.data.loc[:, (column_names, metrics)]], axis=1)
        data = data.reset_index(drop=True)

        res.data = data
        res.metrics = [metric for metric in self.metrics if metric.column_name in metrics]
        res.column_names = [c for c in self.column_names if c in column_names]

        return res


class AbstractCalculator(ABC):
    """Base class for drift calculation."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_number: Optional[int] = None,
        chunk_period: Optional[str] = None,
        chunker: Optional[Chunker] = None,
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
        chunker: Chunker
            The `Chunker` used to split the data sets into a lists of chunks.
        timestamp_column_name: str
            The column name of the column containing timestamp information.
        """
        self.chunker = ChunkerFactory.get_chunker(
            chunk_size, chunk_number, chunk_period, chunker, timestamp_column_name
        )

        self.timestamp_column_name = timestamp_column_name

        self.result: Optional[Result] = None

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> Self:
        """Trains the calculator using reference data."""
        try:
            self._logger.debug(f"fitting {str(self)}")
            return self._fit(reference_data, *args, **kwargs)
        except NannyMLException:
            raise
        except Exception as exc:
            raise CalculatorException(f"failed while fitting {str(self)}.\n{exc}")

    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> Result:
        """Performs a calculation on the provided data."""
        try:
            self._logger.debug(f"calculating {str(self)}")
            return self._calculate(data, *args, **kwargs)
        except NannyMLException:
            raise
        except Exception as exc:
            raise CalculatorException(f"failed while calculating {str(self)}.\n{exc}")

    @abstractmethod
    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> Self:
        raise NotImplementedError(f"'{self.__class__.__name__}' must implement the '_fit' method")

    @abstractmethod
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> Result:
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

    @property
    def empty(self) -> bool:
        return self.data is None or self.data.empty

    def to_df(self, multilevel: bool = True):
        """Export results do pandas dataframe."""
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

    def filter(self, period: str = 'all', metrics: Optional[Union[str, List[str]]] = None, *args, **kwargs) -> Self:
        """Returns result metric data."""
        if metrics and not isinstance(metrics, (str, list)):
            raise InvalidArgumentsException("metrics value provided is not a valid metric or list of metrics")
        if isinstance(metrics, str):
            metrics = [metrics]
        try:
            return self._filter(period, metrics, *args, **kwargs)
        except NannyMLException:
            raise
        except Exception as exc:
            raise EstimatorException(f"could not read result data: {exc}")

    @abstractmethod
    def _filter(self, period: str, metrics: Optional[List[str]] = None, *args, **kwargs) -> Self:
        raise NotImplementedError

    def plot(self, *args, **kwargs) -> plotly.graph_objects.Figure:
        """Plot drift results."""
        raise NotImplementedError


class AbstractEstimator(ABC):
    """Base class for drift calculation."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_number: Optional[int] = None,
        chunk_period: Optional[str] = None,
        chunker: Optional[Chunker] = None,
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
        timestamp_column_name: str
            The column name of the column containing timestamp information.
        """
        self.chunker = ChunkerFactory.get_chunker(
            chunk_size, chunk_number, chunk_period, chunker, timestamp_column_name
        )
        self.timestamp_column_name = timestamp_column_name

        self.result: Optional[Result] = None

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __str__(self):
        return f'{self.__module__}.{self.__class__.__name__}'

    def fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> Self:
        """Trains the calculator using reference data."""
        try:
            self._logger.info(f"fitting {str(self)}")
            return self._fit(reference_data, *args, **kwargs)
        except NannyMLException:
            raise
        except Exception as exc:
            raise CalculatorException(f"failed while fitting {str(self)}.\n{exc}")

    def estimate(self, data: pd.DataFrame, *args, **kwargs) -> Result:
        """Performs a calculation on the provided data."""
        try:
            self._logger.info(f"estimating {str(self)}")
            return self._estimate(data, *args, **kwargs)
        except NannyMLException:
            raise
        except Exception as exc:
            raise CalculatorException(f"failed while calculating {str(self)}.\n{exc}")

    @abstractmethod
    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> Self:
        raise NotImplementedError(f"'{self.__class__.__name__}' must implement the '_fit' method")

    @abstractmethod
    def _estimate(self, data: pd.DataFrame, *args, **kwargs) -> Result:
        raise NotImplementedError(f"'{self.__class__.__name__}' must implement the '_calculate' method")


def _split_features_by_type(data: pd.DataFrame, feature_column_names: Iterable[str]) -> Tuple[List[str], List[str]]:
    continuous_column_names = [col for col in sorted(feature_column_names) if _column_is_continuous(data[col])]

    categorical_column_names = [col for col in sorted(feature_column_names) if _column_is_categorical(data[col])]

    return continuous_column_names, categorical_column_names


def _column_is_categorical(column: pd.Series) -> bool:
    return column.dtype in ['object', 'string', 'category', 'bool']


@overload
def _remove_nans(data: pd.Series) -> pd.Series:
    ...


@overload
def _remove_nans(data: pd.DataFrame, columns: Optional[Iterable[Union[str, Iterable[str]]]]) -> pd.DataFrame:
    ...


def _remove_nans(
    data: Union[pd.Series, pd.DataFrame], columns: Optional[Iterable[Union[str, Iterable[str]]]] = None
) -> Tuple[pd.DataFrame, ...]:
    """Remove rows with NaN values in the specified columns.

    If no columns are given, drop rows with NaN values in any column. If columns are given, drop rows with NaN values
    in the specified columns. If a set of columns is given, drop rows with NaN values in all of the columns in the set.
    """
    # If no columns are given, drop rows with NaN values in any columns
    if columns is None:
        mask = ~data.isna()
        if isinstance(mask, pd.DataFrame):
            mask = mask.all(axis=1)
    else:
        mask = np.ones(len(data), dtype=bool)
        for column_selector in columns:
            nans = data[column_selector].isna()
            if isinstance(nans, pd.DataFrame):
                nans = nans.all(axis=1)
            mask &= ~nans

    # NaN values have been dropped. Try to infer types again
    return data[mask].reset_index(drop=True).infer_objects()


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
    if missing:
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
            "\tLog-based metrics are not supported for negative target values.\n"
            f"\tCheck '{column.name}' at rows {str(negative_item_indices)}."
        )


def _common_nan_removal_dataframe(data: pd.DataFrame, selected_columns: List[str]) -> Tuple[pd.DataFrame, bool]:
    """
    Remove rows of dataframe containing NaN values on selected columns.

    Parameters
    ----------
    data: pd.DataFrame
        Pandas dataframe containing data.
    selected_columns: List[str]
        List containing the strings of column names

    Returns
    -------
    df:
        Dataframe with rows containing NaN's on selected_columns removed. All columns of original
        dataframe are being returned.
    empty:
        Boolean whether the resulting data are contain any rows (false) or not (true)
    """
    if not set(selected_columns) <= set(data.columns):
        raise InvalidArgumentsException(
            f"Selected columns: {selected_columns} not all present in provided data columns {list(data.columns)}"
        )
    df = data.dropna(axis=0, how='any', inplace=False, subset=selected_columns).reset_index(drop=True).infer_objects()
    empty: bool = df.shape[0] == 0
    return df, empty


def _common_nan_removal_ndarrays(data: Sequence[np.array], selected_columns: List[int]) -> Tuple[pd.DataFrame, bool]:
    """
    Remove rows of numpy arrays containing NaN values on selected columns.

    Parameters
    ----------
    data: Sequence[np.array]
        Sequence containing numpy arrays.
    selected_columns: List[int]
        List containing the indices of column numbers

    Returns
    -------
    df:
        Dataframe with rows containing NaN's on selected_columns removed. The columns of the DataFrame are the
        numpy ndarrays in the same order as the input data.
    empty:
        Boolean whether the resulting data are contain any rows (false) or not (true)
    """
    # Check if all selected_columns indices are valid for the first ndarray
    if not all(col < len(data) for col in selected_columns):
        raise InvalidArgumentsException(
            f"Selected columns: {selected_columns} not all present in provided data columns with shape {data[0].shape}"
        )

    # Convert the numpy ndarrays to a pandas dataframe
    df = pd.DataFrame({f'col_{i}': col for i, col in enumerate(data)})

    # Use the dataframe function to remove NaNs
    selected_columns_names = [df.columns[col] for col in selected_columns]
    result, empty = _common_nan_removal_dataframe(df, selected_columns_names)

    return result, empty


@overload
def common_nan_removal(data: pd.DataFrame, selected_columns: List[str]) -> Tuple[pd.DataFrame, bool]:
    ...


@overload
def common_nan_removal(data: Sequence[np.array], selected_columns: List[int]) -> Tuple[pd.DataFrame, bool]:
    ...


def common_nan_removal(
    data: Union[pd.DataFrame, Sequence[np.array]], selected_columns: Union[List[str], List[int]]
) -> Tuple[pd.DataFrame, bool]:
    """
    Wrapper function to handle both pandas DataFrame and sequences of numpy ndarrays.

    Parameters
    ----------
    data: Union[pd.DataFrame, Sequence[np.array]]
        Pandas dataframe or sequence of numpy ndarrays containing data.
    selected_columns: Union[List[str], List[int]]
        List containing the column names or indices

    Returns
    -------
    result:
        Dataframe with rows containing NaN's on selected columns removed. All columns of original
        dataframe or ndarrays are being returned.
    empty:
        Boolean whether the resulting data contains any rows (false) or not (true)
    """
    if isinstance(data, pd.DataFrame):
        if not all(isinstance(col, str) for col in selected_columns):
            raise TypeError("When data is a pandas DataFrame, selected_columns should be a list of strings.")
        return _common_nan_removal_dataframe(data, selected_columns)  # type: ignore
    elif isinstance(data, Sequence) and all(isinstance(arr, np.ndarray) for arr in data):
        if not all(isinstance(col, int) for col in selected_columns):
            raise TypeError("When data is a sequence of numpy ndarrays, selected_columns should be a list of integers.")
        return _common_nan_removal_ndarrays(data, selected_columns)  # type: ignore
    else:
        raise TypeError("Data should be either a pandas DataFrame or a sequence of numpy ndarrays.")
