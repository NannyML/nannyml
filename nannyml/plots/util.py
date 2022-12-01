#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  #
#  License: Apache Software License 2.0

#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import copy
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from nannyml.exceptions import InvalidArgumentsException


def is_time_based_x_axis(
    start_dates: Optional[Union[np.ndarray, pd.Series]], end_dates: Optional[Union[np.ndarray, pd.Series]]
) -> bool:
    if start_dates is None:
        return False

    all_start_none = not np.any(start_dates) if isinstance(start_dates, np.ndarray) else start_dates.isnull().all()

    if end_dates is None:
        return False

    all_end_none = not np.any(end_dates) if isinstance(end_dates, np.ndarray) else end_dates.isnull().all()

    return not all_start_none and not all_end_none


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
    if is_time_based_x_axis(start_dates, end_dates):
        _start_dates = copy.deepcopy(start_dates)
        _start_dates = np.append(_start_dates, end_dates[-1])  # type: ignore
        return _start_dates, _data
    else:
        _chunk_indexes = copy.deepcopy(chunk_indexes)
        _chunk_indexes = np.append(_chunk_indexes, _chunk_indexes[-1] + 1)  # type: ignore
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
        if d is None:
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
