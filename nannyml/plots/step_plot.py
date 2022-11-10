#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from plotly.graph_objects import Scatter


def step_plot(
    data: Union[np.ndarray, pd.Series],
    name,
    color,
    chunk_start_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    chunk_end_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    chunk_indices: Optional[Union[np.ndarray, pd.Series]] = None,
    plot_markers: bool = True,
    **kwargs,
) -> List[Scatter]:

    from nannyml.plots.figure import _check_and_convert

    data, start_dates, end_dates, indices = _check_and_convert(data, chunk_start_dates, chunk_end_dates, chunk_indices)

    from nannyml.plots.figure import add_artificial_endpoint

    x, data = add_artificial_endpoint(indices, start_dates, end_dates, data)

    scatters = [_get_line_scatter(data, x, name, color, **kwargs)]

    if plot_markers:
        scatters.append(_get_marker_scatter(data, x, name, color, **kwargs))

    return scatters


def _get_line_scatter(
    data: Union[np.ndarray, pd.Series],
    x: Union[np.ndarray, pd.Series],
    name: str,
    color: str,
    line_args: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Scatter:
    if line_args is None:
        line_args = {}

    return Scatter(
        name=name,
        mode='lines',
        x=x,
        y=data,
        line=dict(shape='hv', color=color, width=2, dash=None, **line_args),
        hoverinfo='skip',
        **kwargs,
    )


def _get_marker_scatter(
    data: Union[np.ndarray, pd.Series],
    x: Union[np.ndarray, pd.Series],
    name: str,
    color: str,
    marker_args: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Scatter:
    if marker_args is None:
        marker_args = {}

    x_mid = [x1 + (x2 - x1) / 2 for x1, x2 in _pairwise(x)]

    return Scatter(
        name=name,
        mode='markers',
        x=x_mid,
        y=data,
        # TODO: hover
        marker=dict(color=color, size=5, symbol='circle', **marker_args),
        showlegend=False,
        **kwargs,
    )


def _pairwise(x: np.ndarray):
    it = iter(x)
    x1 = next(it, None)
    for x2 in it:
        yield x1, x2
        x1 = x2
