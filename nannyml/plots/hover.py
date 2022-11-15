#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import re
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

from nannyml.exceptions import InvalidArgumentsException
from nannyml.plots.colors import Colors
from nannyml.plots.util import is_time_based_x_axis


class Hover:
    def __init__(self, template: str):
        self.template = template
        self.custom_column_names: List[str] = []
        self.custom_data: List[np.ndarray] = []

    def add(
        self,
        data: Union[np.ndarray, pd.Series],
        name: Optional[str] = None,
    ) -> None:
        if isinstance(data, np.ndarray):
            if name is None:
                raise InvalidArgumentsException("parameter 'name' is required when 'data' is of type 'np.ndarray'")

        if isinstance(data, pd.Series) and name is None:
            name = data.name
            data = data.to_numpy(dtype='object')

        self.custom_column_names.append(name)  # type: ignore
        self.custom_data.append(data)

    def get_template(self) -> str:
        subbed_template = re.sub(
            r"%{(\w+)}",
            lambda m: f'%{{customdata[{self.custom_column_names.index(m.group(1))}]}}'
            if m.group(1) is not None
            else '',
            self.template,
        )
        return subbed_template

    def get_custom_data(self) -> np.ndarray:
        return np.stack(self.custom_data, axis=-1)


def _render_string(column: Union[np.ndarray, pd.Series], render_func: Callable) -> np.ndarray:
    if isinstance(column, pd.Series):
        column = column.to_numpy()

    return np.vectorize(render_func)(column)


def render_period_string(period_column: Union[np.ndarray, pd.Series]) -> np.ndarray:
    return _render_string(
        period_column,
        lambda x: f'<b style="color:{Colors.BLUE_SKY_CRAYOLA};line-height:60px">Reference</b>'
        if x == 'reference'
        else f'<b style="color:{Colors.INDIGO_PERSIAN};line-height:60px">Analysis</b>',
    )


def render_alert_string(alert_column: Union[np.ndarray, pd.Series]) -> np.ndarray:
    return _render_string(
        alert_column, lambda x: '<span style="color:#AD0000">⚠ <b>Drift detected</b></span>' if x else ""
    )


def render_partial_target_string(partial_target_column: Union[np.ndarray, pd.Series]) -> np.ndarray:
    missing_data_style = ''
    return _render_string(
        partial_target_column,
        lambda p: f'Data: <span {missing_data_style if p >= 0.5 else ""}>'
        f'{"⚠ " if p >= 0.5 else ""} <b>{p*100:.2f}% missing</b>'
        '</span>  &nbsp; &nbsp;',
    )


def render_x_coordinate(
    indices_column: Union[np.ndarray, pd.Series],
    start_dates_column: Union[np.ndarray, pd.Series],
    end_dates_column: Union[np.ndarray, pd.Series],
    date_format: str = '%b-%d-%Y',
) -> np.ndarray:
    if is_time_based_x_axis(start_dates_column, end_dates_column):
        return np.array(
            [
                f'From <b>{s.strftime(date_format)}</b> to <b>{e.strftime(date_format)}</b>'
                for s, e in zip(start_dates_column, end_dates_column)
            ]
        )
    else:
        return _render_string(indices_column, lambda i: f'Chunk index: <b>{i}</b>')
