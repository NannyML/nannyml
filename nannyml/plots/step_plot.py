#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from plotly.graph_objects import Figure

from nannyml.plots.hover import Hover
from nannyml.plots.util import add_artificial_endpoint, check_and_convert


def metric(
    figure: Figure,
    data: Union[np.ndarray, pd.Series],
    name,
    color,
    chunk_start_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    chunk_end_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    chunk_indices: Optional[Union[np.ndarray, pd.Series]] = None,
    hover: Optional[Hover] = None,
    plot_markers: bool = True,
    **kwargs,
) -> Figure:
    data, start_dates, end_dates, indices = check_and_convert(data, chunk_start_dates, chunk_end_dates, chunk_indices)
    x, data = add_artificial_endpoint(indices, start_dates, end_dates, data)

    figure = _add_metric_line(figure, data, x, name, color, **kwargs)
    if plot_markers:
        if hover is not None:
            kwargs['hovertemplate'] = hover.get_template()
            kwargs['customdata'] = hover.get_custom_data()
        figure = _add_metric_markers(figure, data, x, name, color, **kwargs)

    return figure


def alert(
    figure: Figure,
    data: Union[np.ndarray, pd.Series],
    alerts: Union[np.ndarray, pd.Series],
    name,
    color,
    chunk_start_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    chunk_end_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    chunk_indices: Optional[Union[np.ndarray, pd.Series]] = None,
    plot_areas: bool = True,
    **kwargs,
) -> Figure:
    data, start_dates, end_dates, indices = check_and_convert(data, chunk_start_dates, chunk_end_dates, chunk_indices)
    x, data = add_artificial_endpoint(indices, start_dates, end_dates, data)

    if isinstance(alerts, pd.Series):
        alerts = alerts.to_numpy()
    alerts = np.append(alerts, alerts[-1])

    figure = _add_alert_markers(figure, data, alerts, x, name, color, **kwargs)
    if plot_areas:
        figure = _add_alert_areas(figure, data, alerts, x, name, color, **kwargs)

    return figure


def _add_metric_line(
    figure: Figure,
    data: Union[np.ndarray, pd.Series],
    x: Union[np.ndarray, pd.Series],
    name: str,
    color: str,
    line_args: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Figure:
    if line_args is None:
        line_args = {}

    figure.add_scatter(
        name=name,
        mode='lines',
        x=x,
        y=data,
        line=dict(shape='hv', color=color, width=2, dash=None, **line_args),
        hoverinfo='skip',
        **kwargs,
    )
    return figure


def _add_metric_markers(
    figure: Figure,
    data: Union[np.ndarray, pd.Series],
    x: Union[np.ndarray, pd.Series],
    name: str,
    color: str,
    marker_args: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Figure:
    if marker_args is None:
        marker_args = {}

    x_mid = [x1 + (x2 - x1) / 2 for x1, x2 in _pairwise(x)]

    figure.add_scatter(
        name=name,
        mode='markers',
        x=x_mid,
        y=data,
        # TODO: hover
        marker=dict(color=color, size=5, symbol='circle', **marker_args),
        showlegend=False,
        **kwargs,
    )

    return figure


def _add_alert_markers(
    figure: Figure,
    data: np.ndarray,
    alerts: np.ndarray,
    x: np.ndarray,
    name: str,
    color: str,
    marker_args: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Figure:
    if marker_args is None:
        marker_args = {}

    alert_indices = [idx for idx, alert in enumerate(alerts) if alert]

    x = x[alert_indices]
    data = data[alert_indices]

    x_mid = [x1 + (x2 - x1) / 2 for x1, x2 in _pairwise(x)]

    figure.add_scatter(
        name=name,
        mode='markers',
        x=x_mid,
        y=data,
        hoverinfo='skip',
        marker=dict(color=color, size=8, symbol='diamond', **marker_args),
        **kwargs,
    )
    return figure


def _add_alert_areas(
    figure: Figure,
    data: np.ndarray,
    alerts: np.ndarray,
    x: np.ndarray,
    name: str,
    color: str,
    marker_args: Optional[Dict[str, Any]] = None,
    alpha: float = 0.2,
    **kwargs,
) -> Figure:
    if marker_args is None:
        marker_args = {}

    alert_indices = [idx for idx, alert in enumerate(alerts) if alert]

    if 'legendgroup' in kwargs:
        del kwargs['legendgroup']

    for x0, x1 in _pairwise(x[alert_indices]):
        figure.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=alpha, layer='below', line_width=0, **kwargs)
    return figure


def _pairwise(x: np.ndarray):
    it = iter(x)
    x1 = next(it, None)
    for x2 in it:
        yield x1, x2
        x1 = x2
