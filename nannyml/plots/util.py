#  Author:  Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

import copy
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from nannyml._typing import TypeGuard
from nannyml.exceptions import InvalidArgumentsException


def has_non_null_data(series: Optional[Union[np.ndarray, pd.Series]]) -> TypeGuard[Union[np.ndarray, pd.Series]]:
    if series is None:
        return False

    return bool(np.any(series) if isinstance(series, np.ndarray) else not series.isnull().all())


def is_time_based_x_axis(
    start_dates: Optional[Union[np.ndarray, pd.Series]], end_dates: Optional[Union[np.ndarray, pd.Series]]
) -> bool:
    return has_non_null_data(start_dates) and has_non_null_data(end_dates)


def add_artificial_endpoint(
    chunk_indexes: Optional[np.ndarray],
    start_dates: Optional[np.ndarray],
    end_dates: Optional[np.ndarray],
    data: Union[np.ndarray, List[np.ndarray]],
):
    _data = copy.deepcopy(data)
    if isinstance(_data, List):
        _data = [np.append(e, e[-1]) for e in data]
    else:
        _data = np.append(_data, _data[-1])
    if has_non_null_data(start_dates) and has_non_null_data(end_dates):
        _start_dates = copy.deepcopy(start_dates)
        _start_dates = np.append(_start_dates, end_dates[-1])
        return _start_dates, _data
    else:
        assert chunk_indexes is not None
        _chunk_indexes = copy.deepcopy(chunk_indexes)
        _chunk_indexes = np.append(_chunk_indexes, _chunk_indexes[-1] + 1)
        return _chunk_indexes, _data


def check_and_convert(
    data: Union[Union[np.ndarray, pd.Series], List[Union[np.ndarray, pd.Series]]],
    chunk_start_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    chunk_end_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    chunk_indices: Optional[Union[np.ndarray, pd.Series]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if chunk_start_dates is None and chunk_end_dates is None and chunk_indices is None:
        raise InvalidArgumentsException(
            "please provide either 'chunk_indices' or " "'chunks_start_dates' and 'chunk_end_dates'"
        )

    if chunk_start_dates is not None and chunk_end_dates is None:
        raise InvalidArgumentsException("'chunk_end_dates' should not be None when 'chunk_start_dates' is not None")

    if chunk_start_dates is None and chunk_end_dates is not None:
        raise InvalidArgumentsException("'chunk_start_dates' should not be None when 'chunk_end_dates' is not None")

    if not isinstance(data, List):
        _data = copy.deepcopy(data)
        if isinstance(_data, pd.Series):
            _data = _data.to_numpy()
    else:
        _data = []
        for d in data:
            _d = copy.deepcopy(d)
            if isinstance(_d, pd.Series):
                _d = _d.to_numpy()
            _data.append(_d)

    if is_time_based_x_axis(chunk_start_dates, chunk_end_dates):
        _start_dates = copy.deepcopy(chunk_start_dates)
        if isinstance(_start_dates, pd.Series):
            _start_dates = _start_dates.to_numpy(dtype=object)

        _end_dates = copy.deepcopy(chunk_end_dates)
        if isinstance(_end_dates, pd.Series):
            _end_dates = _end_dates.to_numpy(dtype=object)

        return _data, _start_dates, _end_dates, None

    else:
        _chunk_indices = copy.deepcopy(chunk_indices)
        if isinstance(_chunk_indices, pd.Series):
            _chunk_indices = _chunk_indices.to_numpy()

        return _data, None, None, _chunk_indices


def ensure_numpy(*args) -> Tuple:
    converted: List[Optional[np.ndarray]] = []
    for d in args:
        if d is None or len(d) == 0:
            converted.append(None)
        elif isinstance(d, pd.Series):
            converted.append(d.to_numpy(dtype='object'))
        elif isinstance(d, np.ndarray):
            converted.append(d)
        else:
            raise InvalidArgumentsException(f"could not convert type '{type(d)}' to 'np.ndarray'")
    return tuple(converted)


def pairwise(x: np.ndarray):
    return [(x[i], x[i + 1]) for i in range(len(x) - 1)]
