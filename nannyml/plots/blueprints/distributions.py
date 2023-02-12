#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import math
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from nannyml.chunk import Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.plots import Colors
from nannyml.plots.components import Figure, Hover
from nannyml.plots.components.joy_plot import alert as joy_alert
from nannyml.plots.components.joy_plot import calculate_chunk_distributions, joy
from nannyml.plots.components.stacked_bar_plot import alert as stacked_bar_alert
from nannyml.plots.components.stacked_bar_plot import calculate_value_counts, stacked_bar
from nannyml.plots.util import is_time_based_x_axis


# Intended to work only with Univariate Drift Result for now...
def plot_distributions(
    result,
    reference_data: pd.DataFrame,  # TODO: move distribution calculations to calculator run
    analysis_data: pd.DataFrame,  # TODO: move distribution calculations to calculator run
    chunker: Chunker,  # TODO: move distribution calculations to calculator run
    title: Optional[str] = 'Column distributions',
    figure: Optional[Figure] = None,
    x_axis_time_title: str = 'Time',
    x_axis_chunk_title: str = 'Chunk',
    y_axis_title: str = 'Values',
    figure_args: Optional[Dict[str, Any]] = None,
    subplot_title_format: str = '<b>{display_names[0]}</b> distribution (alerts for <b>{display_names[1]})</b>',
    number_of_columns: Optional[int] = None,
) -> Figure:
    from nannyml.drift.univariate import Result

    assert isinstance(result, Result)

    number_of_plots = len(result.keys())
    if number_of_columns is None:
        number_of_columns = min(number_of_plots, 1)
    number_of_rows = math.ceil(number_of_plots / number_of_columns)

    if figure_args is None:
        figure_args = {}

    if figure is None:
        figure = Figure(
            **dict(
                title=title,
                x_axis_title=x_axis_time_title
                if is_time_based_x_axis(result.chunk_start_dates, result.chunk_end_dates)
                else x_axis_chunk_title,
                y_axis_title=y_axis_title,
                legend=dict(traceorder="grouped", itemclick=False, itemdoubleclick=False),
                height=number_of_plots * 500 / number_of_columns,
                subplot_args=dict(
                    cols=number_of_columns,
                    rows=number_of_rows,
                    subplot_titles=[
                        subplot_title_format.format(display_names=key.display_names) for key in result.keys()
                    ],
                ),
                **figure_args,
            )
        )

    reference_result = result.filter(period='reference')
    analysis_result = result.filter(period='analysis')

    for idx, key in enumerate(result.keys()):
        row = (idx // number_of_columns) + 1
        col = (idx % number_of_columns) + 1

        column_name, method = key.properties

        analysis_chunk_start_dates = analysis_result.chunk_start_dates
        analysis_chunk_end_dates = analysis_result.chunk_end_dates
        x_axis_is_time_based = is_time_based_x_axis(analysis_chunk_start_dates, analysis_chunk_end_dates)

        if column_name in result.categorical_column_names and method in result.categorical_method_names:
            figure = _plot_stacked_bar(
                figure=figure,
                row=row,
                col=col,
                chunker=chunker,
                column_name=column_name,
                metric_display_name=method,
                reference_data=reference_data[column_name],
                reference_data_timestamps=reference_data[result.timestamp_column_name]
                if x_axis_is_time_based
                else None,
                reference_alerts=reference_result.alerts(key),
                reference_chunk_keys=reference_result.chunk_keys,
                reference_chunk_periods=reference_result.chunk_periods,
                reference_chunk_indices=reference_result.chunk_indices,
                reference_chunk_start_dates=reference_result.chunk_start_dates,
                reference_chunk_end_dates=reference_result.chunk_end_dates,
                analysis_data=analysis_data[column_name],
                analysis_data_timestamps=analysis_data[result.timestamp_column_name] if x_axis_is_time_based else None,
                analysis_alerts=analysis_result.alerts(key),
                analysis_chunk_keys=analysis_result.chunk_keys,
                analysis_chunk_periods=analysis_result.chunk_periods,
                analysis_chunk_indices=analysis_result.chunk_indices,
                analysis_chunk_start_dates=analysis_chunk_start_dates,
                analysis_chunk_end_dates=analysis_chunk_end_dates,
            )
        elif column_name in result.continuous_column_names and method in result.continuous_method_names:
            figure = _plot_joyplot(
                figure=figure,
                row=row,
                col=col,
                chunker=chunker,
                metric_display_name=method,
                reference_data=reference_data[column_name],
                reference_data_timestamps=reference_data[result.timestamp_column_name]
                if x_axis_is_time_based
                else None,
                reference_alerts=reference_result.alerts(key),
                reference_chunk_keys=reference_result.chunk_keys,
                reference_chunk_periods=reference_result.chunk_periods,
                reference_chunk_indices=reference_result.chunk_indices,
                reference_chunk_start_dates=reference_result.chunk_start_dates,
                reference_chunk_end_dates=reference_result.chunk_end_dates,
                analysis_data=analysis_data[column_name],
                analysis_data_timestamps=analysis_data[result.timestamp_column_name] if x_axis_is_time_based else None,
                analysis_alerts=analysis_result.alerts(key),
                analysis_chunk_keys=analysis_result.chunk_keys,
                analysis_chunk_periods=analysis_result.chunk_periods,
                analysis_chunk_indices=analysis_result.chunk_indices,
                analysis_chunk_start_dates=analysis_chunk_start_dates,
                analysis_chunk_end_dates=analysis_chunk_end_dates,
            )
        else:
            raise InvalidArgumentsException(
                f"can not plot column '{column_name}' since the column is not in "
                f"the continuous or categorical columns lists."
            )
    return figure


def _plot_joyplot(
    figure: Figure,
    metric_display_name: str,
    reference_data: Union[np.ndarray, pd.Series],
    reference_data_timestamps: Union[np.ndarray, pd.Series],
    analysis_data: Union[np.ndarray, pd.Series],
    analysis_data_timestamps: Union[np.ndarray, pd.Series],
    chunker: Chunker,
    reference_alerts: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_keys: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_periods: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_indices: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_start_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_end_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_alerts: Optional[Union[np.ndarray, pd.Series]] = None,
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
    subplot_args = dict(row=row, col=col) if is_subplot else None

    has_reference_results = reference_chunk_indices is not None and len(reference_chunk_indices) > 0

    if figure is None:
        figure = Figure(title='continuous distribution', x_axis_title='time', y_axis_title='value', height=500)

    if has_reference_results:  # TODO: move distribution calculations to calculator run
        reference_distributions = calculate_chunk_distributions(
            data=reference_data,
            chunker=chunker,
            data_periods=pd.Series('reference', index=range(len(reference_data))),
            timestamps=reference_data_timestamps,
        )

        figure = joy(
            fig=figure,
            data_distributions=reference_distributions,
            chunk_indices=reference_chunk_indices,
            chunk_start_dates=reference_chunk_start_dates,
            chunk_end_dates=reference_chunk_end_dates,
            name='Reference',
            color=Colors.BLUE_SKY_CRAYOLA,
            subplot_args=subplot_args,
        )

        assert reference_chunk_indices is not None
        analysis_chunk_indices = analysis_chunk_indices + (max(reference_chunk_indices) + 1)

    analysis_distributions = calculate_chunk_distributions(  # TODO: move distribution calculations to calculator run
        data=analysis_data,
        chunker=chunker,
        data_periods=pd.Series('analysis', index=range(len(analysis_data))),
        timestamps=analysis_data_timestamps,
    )

    figure = joy(
        fig=figure,
        data_distributions=analysis_distributions,
        chunk_indices=analysis_chunk_indices,
        chunk_start_dates=analysis_chunk_start_dates,
        chunk_end_dates=analysis_chunk_end_dates,
        name='Analysis',
        color=Colors.INDIGO_PERSIAN,
        subplot_args=subplot_args,
    )

    if analysis_alerts is not None:
        figure = joy_alert(
            fig=figure,
            alerts=analysis_alerts,
            data_distributions=analysis_distributions,
            color=Colors.RED_IMPERIAL,
            name='Alerts',
            chunk_indices=analysis_chunk_indices,
            chunk_start_dates=analysis_chunk_start_dates,
            chunk_end_dates=analysis_chunk_end_dates,
            subplot_args=subplot_args,
        )

    return figure


def _plot_stacked_bar(
    figure: Figure,
    column_name: str,
    metric_display_name: str,
    reference_data: Union[np.ndarray, pd.Series],
    reference_data_timestamps: Union[np.ndarray, pd.Series],
    analysis_data: Union[np.ndarray, pd.Series],
    analysis_data_timestamps: Union[np.ndarray, pd.Series],
    chunker: Chunker,
    reference_alerts: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_keys: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_periods: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_indices: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_start_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_end_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_alerts: Optional[Union[np.ndarray, pd.Series]] = None,
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
    subplot_args = dict(row=row, col=col) if is_subplot else None

    has_reference_results = reference_chunk_indices is not None and len(reference_chunk_indices) > 0

    if figure is None:
        figure = Figure(title='continuous distribution', x_axis_title='time', y_axis_title='value', height=500)

    figure.update_xaxes(
        dict(mirror=False, showline=False),
        overwrite=True,
        matches='x',
        title=figure.layout.xaxis.title,
        row=row,
        col=col,
    )
    figure.update_yaxes(
        dict(mirror=False, showline=False), overwrite=True, title=figure.layout.yaxis.title, row=row, col=col
    )

    if has_reference_results:
        reference_value_counts = calculate_value_counts(
            data=reference_data,
            chunker=chunker,
            timestamps=reference_data_timestamps,
            max_number_of_categories=5,
            missing_category_label='Missing',
        )

        figure = stacked_bar(
            figure=figure,
            stacked_bar_table=reference_value_counts,
            color=Colors.BLUE_SKY_CRAYOLA,
            chunk_indices=reference_chunk_indices,
            chunk_start_dates=reference_chunk_start_dates,
            chunk_end_dates=reference_chunk_end_dates,
            annotation='Reference',
            showlegend=True,
            legendgrouptitle_text=f'<b>{column_name}</b>',
            legendgroup=column_name,
            subplot_args=subplot_args,
        )

        assert reference_chunk_indices is not None
        analysis_chunk_indices = analysis_chunk_indices + (max(reference_chunk_indices) + 1)

    analysis_value_counts = calculate_value_counts(
        data=analysis_data,
        chunker=chunker,
        timestamps=analysis_data_timestamps,
        max_number_of_categories=5,
        missing_category_label='Missing',
    )

    figure = stacked_bar(
        figure=figure,
        stacked_bar_table=analysis_value_counts,
        color=Colors.INDIGO_PERSIAN,
        chunk_indices=analysis_chunk_indices,
        chunk_start_dates=analysis_chunk_start_dates,
        chunk_end_dates=analysis_chunk_end_dates,
        annotation='Analysis',
        showlegend=False,
        legendgroup=column_name,
        subplot_args=subplot_args,
    )

    figure = stacked_bar_alert(
        figure=figure,
        alerts=analysis_alerts,
        stacked_bar_table=analysis_value_counts,
        color=Colors.RED_IMPERIAL,
        chunk_indices=analysis_chunk_indices,
        chunk_start_dates=analysis_chunk_start_dates,
        chunk_end_dates=analysis_chunk_end_dates,
        showlegend=True,
        legendgroup=column_name,
        subplot_args=subplot_args,
    )

    return figure
