#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import copy
import math
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from nannyml.chunk import Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.plots import Colors, ensure_numpy, is_time_based_x_axis
from nannyml.plots.components import Figure, Hover, render_alert_string, render_period_string, render_x_coordinate
from nannyml.plots.components.joy_plot import calculate_chunk_distributions, joy, alert


def plot_continuous_distribution_list(
    result,
    title: str,
    x_axis_time_title: str = 'Time',
    x_axis_chunk_title: str = 'Chunk',
    y_axis_title: str = 'Metric',
    figure_args: Optional[Dict[str, Any]] = None,
    subplot_title_format: str = 'Metric <b>{metric_name}</b>',
) -> Figure:
    number_of_columns = min(len(result.metrics), 2)

    reference_result: pd.DataFrame = result.filter(period='reference').to_df()
    analysis_result: pd.DataFrame = result.filter(period='analysis').to_df()

    if figure_args is None:
        figure_args = {}

    figure = Figure(
        **dict(
            title=title,
            x_axis_title=x_axis_time_title if result.timestamp_column_name else x_axis_chunk_title,
            y_axis_title=y_axis_title,
            legend=dict(traceorder="grouped", itemclick=False, itemdoubleclick=False),
            height=len(result.metrics) * 500 / number_of_columns,
            subplot_args=dict(
                cols=number_of_columns,
                rows=math.ceil(len(result.metrics) / number_of_columns),
                subplot_titles=[
                    subplot_title_format.format(metric_name=metric.display_name) for metric in result.metrics
                ],
            ),
            **figure_args,
        )
    )

    for idx, metric in enumerate(result.metrics):
        row = (idx // number_of_columns) + 1
        col = (idx % number_of_columns) + 1

        figure = _plot_joyplot(
            figure=figure,
            row=row,
            col=col,

            metric_display_name=metric.display_name,
            reference_metric=reference_result[(metric.column_name, 'value')],
            reference_alerts=reference_result.get((metric.column_name, 'alert'), default=None),
            reference_chunk_keys=reference_result.get(('chunk', 'key'), default=None),
            reference_chunk_periods=reference_result.get(('chunk', 'period'), default=None),
            reference_chunk_indices=reference_result.get(('chunk', 'chunk_index'), default=None),
            reference_chunk_start_dates=reference_result.get(('chunk', 'start_date'), default=None),
            reference_chunk_end_dates=reference_result.get(('chunk', 'end_date'), default=None),
            analysis_metric=analysis_result[(metric.column_name, 'value')],
            analysis_alerts=analysis_result.get((metric.column_name, 'alert'), default=None),
            analysis_chunk_keys=analysis_result.get(('chunk', 'key'), default=None),
            analysis_chunk_periods=analysis_result.get(('chunk', 'period'), default=None),
            analysis_chunk_indices=analysis_result.get(('chunk', 'chunk_index'), default=None),
            analysis_chunk_start_dates=analysis_result.get(('chunk', 'start_date'), default=None),
            analysis_chunk_end_dates=analysis_result.get(('chunk', 'end_date'), default=None),
        )

    return figure


def _plot_joyplot(
    figure: Figure,
    metric_display_name: str,
    analysis_metric: Union[np.ndarray, pd.Series],
    chunker: Chunker,
    reference_metric: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_alerts: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_timestamps: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_keys: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_periods: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_indices: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_start_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_end_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_alerts: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_timestamps: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_chunk_keys: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_chunk_periods: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_chunk_indices: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_chunk_start_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_chunk_end_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    row: Optional[int] = None,
    col: Optional[int] = None,
    hover: Optional[Hover] = None,
) -> Figure:
    is_subplot = row is not None and col is not None

    show_in_legend = row == 1 and col == 1 if is_subplot else True

    has_reference_results = reference_chunk_indices is not None and len(reference_chunk_indices) > 0

    if figure is None:
        figure = Figure(title='continuous distribution', x_axis_title='time', y_axis_title='value', height=500)

    if has_reference_results:
        reference_distributions = calculate_chunk_distributions(
            data=reference_metric,
            chunker=chunker,
            data_periods=reference_chunk_periods,
            timestamps=reference_timestamps
        )

        figure = joy(
            fig=figure,
            data_distributions=reference_distributions,
            chunk_indices=reference_chunk_indices,
            chunk_start_dates=reference_chunk_start_dates,
            chunk_end_dates=reference_chunk_end_dates,
            name='Reference',
            color=Colors.BLUE_SKY_CRAYOLA
        )

    analysis_distributions = calculate_chunk_distributions(
        data=analysis_metric,
        chunker=chunker,
        data_periods=analysis_chunk_periods,
        timestamps=analysis_timestamps,
    )

    figure = joy(
        fig=figure,
        data_distributions=analysis_distributions,
        chunk_indices=analysis_chunk_indices,
        chunk_start_dates=analysis_chunk_start_dates,
        chunk_end_dates=analysis_chunk_end_dates,
        name='Reference',
        color=Colors.INDIGO_PERSIAN
    )

    if analysis_alerts is not None:
        figure = alert(
            fig=figure,
            alerts=analysis_alerts,
            data_distributions=analysis_distributions,
            color=Colors.RED_IMPERIAL,
            name='Alerts',
            chunk_indices=analysis_chunk_indices,
            chunk_start_dates=analysis_chunk_start_dates,
            chunk_end_dates=analysis_chunk_end_dates,
        )

    return figure



