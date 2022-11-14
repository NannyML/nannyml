#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import copy
from typing import List, Optional, Tuple, Union

import matplotlib.colors
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from nannyml.exceptions import InvalidArgumentsException
from nannyml.plots.colors import Colors
from nannyml.plots.step_plot import alert as step_plot_alert
from nannyml.plots.step_plot import metric as step_plot_metric


class Figure(go.Figure):
    """Extending the Plotly Figure class functionality."""

    SUPPORTED_METRIC_STYLES = ['step']

    def __init__(
        self,
        title: str,
        x_axis_title: str,
        y_axis_title: str,
        y_axis_limit: Optional[List] = None,
        metric_style: str = 'step',
        **kwargs,
    ):
        """Creates a new Figure."""

        layout = go.Layout(
            title=title,
            xaxis=dict(
                title=x_axis_title, linecolor=Colors.INDIGO_PERSIAN, showgrid=False, mirror=True, zeroline=False
            ),
            yaxis=dict(
                title=y_axis_title,
                linecolor=Colors.INDIGO_PERSIAN,
                showgrid=False,
                range=y_axis_limit,
                mirror=True,
                zeroline=False,
            ),
            paper_bgcolor='rgba(255,255,255,1)',
            plot_bgcolor='rgba(255,255,255,1)',
            hoverlabel=dict(bgcolor="white", font_size=14),
            **kwargs,
        )
        super().__init__(layout=layout)

        if metric_style not in self.SUPPORTED_METRIC_STYLES:
            raise InvalidArgumentsException(
                f"metric style '{metric_style}' is not supported. "
                "Please provide one of the following values: "
                f"{str.join(',', self.SUPPORTED_METRIC_STYLES)}"
            )
        self._metric_style: str = metric_style

    def add_metric(
        self,
        data: Union[np.ndarray, pd.Series],
        name: str,
        color: str,
        indices: Optional[Union[np.ndarray, pd.Series]] = None,
        start_dates: Optional[Union[np.ndarray, pd.Series]] = None,
        end_dates: Optional[Union[np.ndarray, pd.Series]] = None,
        **kwargs,
    ):
        if self._metric_style not in self.SUPPORTED_METRIC_STYLES:
            raise InvalidArgumentsException(
                f"metric style '{self._metric_style}' is not supported. "
                "Please provide one of the following values: "
                f"{str.join(',', self.SUPPORTED_METRIC_STYLES)}"
            )

        if self._metric_style == 'step':
            step_plot_metric(
                figure=self,
                data=data,
                chunk_indices=indices,
                chunk_start_dates=start_dates,
                chunk_end_dates=end_dates,
                name=name,
                color=color,
                **kwargs,
            )

    def add_period_separator(self, x, color=Colors.BLUE_SKY_CRAYOLA):
        self.add_vline(
            x=x,
            line=dict(color=color, width=1, dash='dash'),
            layer='below',
        )

    def add_threshold(
        self,
        data,
        name,
        indices=None,
        start_dates=None,
        end_dates=None,
        color=Colors.RED_IMPERIAL,
        with_additional_endpoint: bool = False,
        **kwargs,
    ):
        data, start_dates, end_dates, indices = _check_and_convert(data, start_dates, end_dates, indices)
        x = start_dates if is_time_based_x_axis(start_dates, end_dates) else indices

        if with_additional_endpoint:
            x, data = add_artificial_endpoint(indices, start_dates, end_dates, data)

        self.add_trace(
            go.Scatter(
                name=name,
                mode='lines',
                x=x,
                y=data,
                line=dict(color=color, width=2, dash='dash'),
                hoverinfo='skip',
                **kwargs,
            )
        )

    def add_confidence_band(
        self,
        upper_confidence_boundaries,
        lower_confidence_boundaries,
        name,
        indices=None,
        start_dates=None,
        end_dates=None,
        color=Colors.RED_IMPERIAL,
        with_additional_endpoint: bool = False,
        **kwargs,
    ):
        data, start_dates, end_dates, indices = _check_and_convert(
            [upper_confidence_boundaries, lower_confidence_boundaries], start_dates, end_dates, indices
        )
        x = start_dates if is_time_based_x_axis(start_dates, end_dates) else indices

        if with_additional_endpoint:
            x, data = add_artificial_endpoint(indices, start_dates, end_dates, data)

        self.add_traces(
            [
                go.Scatter(
                    name=name,
                    mode='lines',
                    x=x,
                    y=data[0],
                    line=dict(shape='hv', color='rgba(0,0,0,0)'),
                    hoverinfo='skip',
                    showlegend=False,
                    **kwargs,
                ),
                go.Scatter(
                    name=name,
                    mode='lines',
                    x=x,
                    y=data[1],
                    line=dict(shape='hv', color='rgba(0,0,0,0)'),
                    fill='tonexty',
                    fillcolor='rgba{}'.format(matplotlib.colors.to_rgba(matplotlib.colors.to_rgb(color), alpha=0.2)),
                    hoverinfo='skip',
                    showlegend=True,
                    **kwargs,
                ),
            ]
        )

    def add_alert(
        self,
        data: Union[np.ndarray, pd.Series],
        name: str,
        color: str = Colors.RED_IMPERIAL,
        indices: Optional[Union[np.ndarray, pd.Series]] = None,
        start_dates: Optional[Union[np.ndarray, pd.Series]] = None,
        end_dates: Optional[Union[np.ndarray, pd.Series]] = None,
        **kwargs,
    ):
        if self._metric_style not in self.SUPPORTED_METRIC_STYLES:
            raise InvalidArgumentsException(
                f"metric style '{self._metric_style}' is not supported. "
                "Please provide one of the following values: "
                f"{str.join(',', self.SUPPORTED_METRIC_STYLES)}"
            )

        if self._metric_style == 'step':
            step_plot_alert(
                figure=self,
                data=data,
                chunk_indices=indices,
                chunk_start_dates=start_dates,
                chunk_end_dates=end_dates,
                name=name,
                color=color,
                **kwargs,
            )


def is_time_based_x_axis(
    start_dates: Optional[Union[np.ndarray, pd.Series]], end_dates: Optional[Union[np.ndarray, pd.Series]]
) -> bool:
    return start_dates is not None and end_dates is not None


def add_artificial_endpoint(
    chunk_indexes: np.ndarray,
    start_dates: np.ndarray,
    end_dates: np.ndarray,
    data: np.ndarray,
):
    _data = copy.deepcopy(data)
    _data = np.append(_data, _data[-1])
    if is_time_based_x_axis(start_dates, end_dates):
        _start_dates = copy.deepcopy(start_dates)
        _start_dates = np.append(_start_dates, end_dates[-1])
        return _start_dates, _data
    else:
        _chunk_indexes = copy.deepcopy(chunk_indexes)
        _chunk_indexes = np.append(_chunk_indexes, _chunk_indexes[-1] + 1)
        return _chunk_indexes, _data


def _check_and_convert(
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
        if isinstance(data, pd.Series):
            _data = _data.to_numpy()
    else:
        _data = []
        for d in data:
            _d = copy.deepcopy(d)
            if isinstance(data, pd.Series):
                _d = _d.to_numpy()
            _data.append(_d)

    if chunk_start_dates is not None and chunk_end_dates is not None:
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
