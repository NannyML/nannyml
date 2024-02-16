#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from typing import Any, Dict, List, Optional, Union

import matplotlib.colors
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from nannyml.exceptions import InvalidArgumentsException
from nannyml.plots.colors import Colors
from nannyml.plots.components.hover import Hover
from nannyml.plots.components.step_plot import alert as step_plot_alert
from nannyml.plots.components.step_plot import metric as step_plot_metric
from nannyml.plots.util import add_artificial_endpoint, check_and_convert, is_time_based_x_axis


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
        subplot_args: Optional[Dict[str, Any]] = None,
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

        if subplot_args is not None:
            self.set_subplots(**subplot_args)

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
        hover: Optional[Hover] = None,
        subplot_args: Optional[Dict[str, Any]] = None,
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
                hover=hover,
                subplot_args=subplot_args,
                **kwargs,
            )

    def add_period_separator(self, x, color=Colors.GRAY_DARK, subplot_args: Optional[Dict[str, Any]] = None, **kwargs):
        if subplot_args is not None:
            kwargs.update(dict(col=subplot_args.get('col'), row=subplot_args.get('row')))

        self.add_vline(x=x, line=dict(color=color, width=1), layer='below', **kwargs)

    def add_threshold(
        self,
        data,
        name,
        indices=None,
        start_dates=None,
        end_dates=None,
        color=Colors.RED_IMPERIAL,
        with_additional_endpoint: bool = False,
        subplot_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        data, start_dates, end_dates, indices = check_and_convert(data, start_dates, end_dates, indices)
        x = start_dates if is_time_based_x_axis(start_dates, end_dates) else indices

        if with_additional_endpoint:
            x, data = add_artificial_endpoint(indices, start_dates, end_dates, data)

        show_in_legend = True
        if subplot_args is None:
            subplot_args = {}
        else:
            if 'showlegend' in kwargs:
                is_first_subplot = subplot_args.get('row', 0) == 1 and subplot_args.get('col', 0) == 1
                show_in_legend = kwargs['showlegend'] and is_first_subplot
        kwargs['showlegend'] = show_in_legend

        self.add_trace(
            go.Scatter(
                name=name,
                mode='lines',
                x=x,
                y=data,
                line=dict(color=color, width=2, dash='dash'),
                hoverinfo='skip',
                **kwargs,
            ),
            **subplot_args,
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
        subplot_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        data, start_dates, end_dates, indices = check_and_convert(
            [upper_confidence_boundaries, lower_confidence_boundaries], start_dates, end_dates, indices
        )
        x = start_dates if is_time_based_x_axis(start_dates, end_dates) else indices

        if with_additional_endpoint:
            x, data = add_artificial_endpoint(indices, start_dates, end_dates, data)

        show_in_legend = kwargs.get('showlegend', True)
        is_single_plot = subplot_args and subplot_args.get('row') is None and subplot_args.get('col') is None

        if subplot_args is None or is_single_plot:
            subplot_args = {}

        del kwargs['showlegend']

        self.add_trace(
            go.Scatter(
                name=name,
                mode='lines',
                x=x,
                y=data[0],
                line=dict(shape='hv', color='rgba(0,0,0,0)'),
                hoverinfo='skip',
                showlegend=False,
                connectgaps=True,
                **kwargs,
            ),
            **subplot_args,
        )
        self.add_trace(
            go.Scatter(
                name=name,
                mode='lines',
                x=x,
                y=data[1],
                line=dict(shape='hv', color='rgba(0,0,0,0)'),
                fill='tonexty',
                fillcolor='rgba{}'.format(matplotlib.colors.to_rgba(matplotlib.colors.to_rgb(color), alpha=0.2)),
                hoverinfo='skip',
                showlegend=show_in_legend,
                connectgaps=True,
                **kwargs,
            ),
            **subplot_args,
        )

    def add_alert(
        self,
        data: Union[np.ndarray, pd.Series],
        name: str,
        color: str = Colors.RED_IMPERIAL,
        indices: Optional[Union[np.ndarray, pd.Series]] = None,
        start_dates: Optional[Union[np.ndarray, pd.Series]] = None,
        end_dates: Optional[Union[np.ndarray, pd.Series]] = None,
        subplot_args: Optional[Dict[str, Any]] = None,
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
                subplot_args=subplot_args,
                **kwargs,
            )
