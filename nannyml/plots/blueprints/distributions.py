#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import math
from typing import Any, Dict, List, Optional, Tuple, Union

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
def plot_2d_univariate_distributions_list(
    result,
    reference_data: pd.DataFrame,  # TODO: move distribution calculations to calculator run
    analysis_data: pd.DataFrame,  # TODO: move distribution calculations to calculator run
    chunker: Chunker,  # TODO: move distribution calculations to calculator run
    items: List[Tuple[str, Any]],
    title: Optional[str] = 'Column distributions',
    figure: Optional[Figure] = None,
    x_axis_time_title: str = 'Time',
    x_axis_chunk_title: str = 'Chunk',
    y_axis_title: str = 'Values',
    figure_args: Optional[Dict[str, Any]] = None,
    subplot_title_format: str = '<b>{column_name}</b> distribution (alerts for <b>{method_name})</b>',
    number_of_columns: Optional[int] = None,
) -> Figure:
    if not items:
        raise InvalidArgumentsException("tried plotting distributions but received zero plotting items.")
    number_of_plots = len(items)
    if number_of_columns is None:
        number_of_columns = min(number_of_plots, 1)
    number_of_rows = math.ceil(number_of_plots / number_of_columns)

    if figure_args is None:
        figure_args = {}

    if figure is None:
        figure = Figure(
            **dict(
                title=title,
                x_axis_title=x_axis_time_title if result.timestamp_column_name else x_axis_chunk_title,
                y_axis_title=y_axis_title,
                legend=dict(traceorder="grouped", itemclick=False, itemdoubleclick=False),
                height=number_of_plots * 500 / number_of_columns,
                subplot_args=dict(
                    cols=number_of_columns,
                    rows=number_of_rows,
                    subplot_titles=[
                        subplot_title_format.format(column_name=column_name, method_name=method.display_name)
                        for column_name, method in items
                    ],
                ),
                **figure_args,
            )
        )

    reference_result: pd.DataFrame = result.filter(period='reference').to_df()
    analysis_result: pd.DataFrame = result.filter(period='analysis').to_df()

    for idx, (column_name, method) in enumerate(items):
        row = (idx // number_of_columns) + 1
        col = (idx % number_of_columns) + 1

        analysis_chunk_start_dates = analysis_result.get(('chunk', 'chunk', 'start_date'), default=None)
        analysis_chunk_end_dates = analysis_result.get(('chunk', 'chunk', 'end_date'), default=None)
        x_axis_is_time_based = is_time_based_x_axis(analysis_chunk_start_dates, analysis_chunk_end_dates)

        if column_name in result.categorical_column_names and method in result.categorical_methods:
            figure = _plot_stacked_bar(
                figure=figure,
                row=row,
                col=col,
                chunker=chunker,
                column_name=column_name,
                metric_display_name=method.display_name,
                reference_data=reference_data[column_name],
                reference_data_timestamps=reference_data[result.timestamp_column_name]
                if x_axis_is_time_based
                else None,
                reference_alerts=reference_result.get((column_name, method.column_name, 'alert'), default=None),
                reference_chunk_keys=reference_result.get(('chunk', 'chunk', 'key'), default=None),
                reference_chunk_periods=reference_result.get(('chunk', 'chunk', 'period'), default=None),
                reference_chunk_indices=reference_result.get(('chunk', 'chunk', 'chunk_index'), default=None),
                reference_chunk_start_dates=reference_result.get(('chunk', 'chunk', 'start_date'), default=None),
                reference_chunk_end_dates=reference_result.get(('chunk', 'chunk', 'end_date'), default=None),
                analysis_data=analysis_data[column_name],
                analysis_data_timestamps=analysis_data[result.timestamp_column_name] if x_axis_is_time_based else None,
                analysis_alerts=analysis_result.get((column_name, method.column_name, 'alert'), default=None),
                analysis_chunk_keys=analysis_result.get(('chunk', 'chunk', 'key'), default=None),
                analysis_chunk_periods=analysis_result.get(('chunk', 'chunk', 'period'), default=None),
                analysis_chunk_indices=analysis_result.get(('chunk', 'chunk', 'chunk_index'), default=None),
                analysis_chunk_start_dates=analysis_chunk_start_dates,
                analysis_chunk_end_dates=analysis_chunk_end_dates,
            )
        elif column_name in result.continuous_column_names and method in result.continuous_methods:
            figure = _plot_joyplot(
                figure=figure,
                row=row,
                col=col,
                chunker=chunker,
                metric_display_name=method.display_name,
                reference_data=reference_data[column_name],
                reference_data_timestamps=reference_data[result.timestamp_column_name]
                if x_axis_is_time_based
                else None,
                reference_alerts=reference_result.get((column_name, method.column_name, 'alert'), default=None),
                reference_chunk_keys=reference_result.get(('chunk', 'chunk', 'key'), default=None),
                reference_chunk_periods=reference_result.get(('chunk', 'chunk', 'period'), default=None),
                reference_chunk_indices=reference_result.get(('chunk', 'chunk', 'chunk_index'), default=None),
                reference_chunk_start_dates=reference_result.get(('chunk', 'chunk', 'start_date'), default=None),
                reference_chunk_end_dates=reference_result.get(('chunk', 'chunk', 'end_date'), default=None),
                analysis_data=analysis_data[column_name],
                analysis_data_timestamps=analysis_data[result.timestamp_column_name] if x_axis_is_time_based else None,
                analysis_alerts=analysis_result.get((column_name, method.column_name, 'alert'), default=None),
                analysis_chunk_keys=analysis_result.get(('chunk', 'chunk', 'key'), default=None),
                analysis_chunk_periods=analysis_result.get(('chunk', 'chunk', 'period'), default=None),
                analysis_chunk_indices=analysis_result.get(('chunk', 'chunk', 'chunk_index'), default=None),
                analysis_chunk_start_dates=analysis_result.get(('chunk', 'chunk', 'start_date'), default=None),
                analysis_chunk_end_dates=analysis_result.get(('chunk', 'chunk', 'end_date'), default=None),
            )
        else:
            raise InvalidArgumentsException(
                f"can not plot column '{column_name}' since the column is not in "
                f"the continuous or categorical columns lists."
            )
    return figure


def plot_2d_continuous_distribution_list(
    result,
    reference_data: pd.DataFrame,  # TODO: move distribution calculations to calculator run
    analysis_data: pd.DataFrame,  # TODO: move distribution calculations to calculator run
    chunker: Chunker,  # TODO: move distribution calculations to calculator run
    title: Optional[str] = None,
    figure: Optional[Figure] = None,
    x_axis_time_title: str = 'Time',
    x_axis_chunk_title: str = 'Chunk',
    y_axis_title: str = 'Metric',
    figure_args: Optional[Dict[str, Any]] = None,
    subplot_title_format: str = 'Metric <b>{metric_name}</b>',
    dimension_1_name: str = 'column_name',
    dimension_2_name: str = 'metric',
) -> Figure:
    dimension_1 = _try_get_dimension(result, dimension_1_name)
    dimension_2 = _try_get_dimension(result, dimension_2_name)

    number_of_plots = len(dimension_1) * len(dimension_2)
    number_of_columns = min(number_of_plots, 2)
    number_of_rows = math.ceil(number_of_plots / number_of_columns)

    if figure_args is None:
        figure_args = {}

    if figure is None:
        figure = Figure(
            **dict(
                title=title,
                x_axis_title=x_axis_time_title if result.timestamp_column_name else x_axis_chunk_title,
                y_axis_title=y_axis_title,
                legend=dict(traceorder="grouped", itemclick=False, itemdoubleclick=False),
                height=number_of_plots * 500 / number_of_columns,
                subplot_args=dict(
                    cols=number_of_columns,
                    rows=number_of_rows,
                    subplot_titles=[
                        subplot_title_format.format(dimension_1=d1_value, dimension_2=d2_value)
                        for d1_value in dimension_1
                        for d2_value in dimension_2
                    ],
                ),
                **figure_args,
            )
        )

    reference_result: pd.DataFrame = result.filter(period='reference').to_df()
    analysis_result: pd.DataFrame = result.filter(period='analysis').to_df()

    for d1_idx, d1_value in enumerate(dimension_1):
        for d2_idx, d2_value in enumerate(dimension_2):
            idx = d1_idx * len(dimension_2) + d2_idx
            row = (idx // number_of_columns) + 1
            col = (idx % number_of_columns) + 1

            figure = _plot_joyplot(
                figure=figure,
                row=row,
                col=col,
                chunker=chunker,
                metric_display_name=d2_value.display_name,
                reference_data=reference_data[d1_value],
                reference_data_timestamps=reference_data[result.timestamp_column_name],
                reference_alerts=reference_result.get((d1_value, d2_value.column_name, 'alert'), default=None),
                reference_chunk_keys=reference_result.get(('chunk', 'chunk', 'key'), default=None),
                reference_chunk_periods=reference_result.get(('chunk', 'chunk', 'period'), default=None),
                reference_chunk_indices=reference_result.get(('chunk', 'chunk', 'chunk_index'), default=None),
                reference_chunk_start_dates=reference_result.get(('chunk', 'chunk', 'start_date'), default=None),
                reference_chunk_end_dates=reference_result.get(('chunk', 'chunk', 'end_date'), default=None),
                analysis_data=analysis_data[d1_value],
                analysis_data_timestamps=analysis_data[result.timestamp_column_name],
                analysis_alerts=analysis_result.get((d1_value, d2_value.column_name, 'alert'), default=None),
                analysis_chunk_keys=analysis_result.get(('chunk', 'chunk', 'key'), default=None),
                analysis_chunk_periods=analysis_result.get(('chunk', 'chunk', 'period'), default=None),
                analysis_chunk_indices=analysis_result.get(('chunk', 'chunk', 'chunk_index'), default=None),
                analysis_chunk_start_dates=analysis_result.get(('chunk', 'chunk', 'start_date'), default=None),
                analysis_chunk_end_dates=analysis_result.get(('chunk', 'chunk', 'end_date'), default=None),
            )

    return figure


def plot_2d_categorical_distribution_list(
    result,
    reference_data: pd.DataFrame,  # TODO: move distribution calculations to calculator run
    analysis_data: pd.DataFrame,  # TODO: move distribution calculations to calculator run
    chunker: Chunker,  # TODO: move distribution calculations to calculator run
    title: Optional[str] = None,
    figure: Optional[Figure] = None,
    x_axis_time_title: str = 'Time',
    x_axis_chunk_title: str = 'Chunk',
    y_axis_title: str = 'Metric',
    figure_args: Optional[Dict[str, Any]] = None,
    subplot_title_format: str = 'Metric <b>{metric_name}</b>',
    dimension_1_name: str = 'column_name',
    dimension_2_name: str = 'metric',
) -> Figure:
    dimension_1 = _try_get_dimension(result, dimension_1_name)
    dimension_2 = _try_get_dimension(result, dimension_2_name)

    number_of_plots = len(dimension_1) * len(dimension_2)
    number_of_columns = min(number_of_plots, 2)
    number_of_rows = math.ceil(number_of_plots / number_of_columns)

    if figure_args is None:
        figure_args = {}

    if figure is None:
        figure = Figure(
            **dict(
                title=title,
                x_axis_title=x_axis_time_title if result.timestamp_column_name else x_axis_chunk_title,
                y_axis_title=y_axis_title,
                legend=dict(traceorder="grouped", itemclick=False, itemdoubleclick=False),
                height=number_of_plots * 500 / number_of_columns,
                subplot_args=dict(
                    cols=number_of_columns,
                    rows=number_of_rows,
                    subplot_titles=[
                        subplot_title_format.format(dimension_1=d1_value, dimension_2=d2_value)
                        for d1_value in dimension_1
                        for d2_value in dimension_2
                    ],
                ),
                **figure_args,
            )
        )
    else:
        figure.set_subplots(rows=6 + number_of_rows, cols=number_of_columns)

    reference_result: pd.DataFrame = result.filter(period='reference').to_df()
    analysis_result: pd.DataFrame = result.filter(period='analysis').to_df()

    for d1_idx, d1_value in enumerate(dimension_1):
        for d2_idx, d2_value in enumerate(dimension_2):
            idx = d1_idx * len(dimension_2) + d2_idx + 6 * 2
            row = (idx // number_of_columns) + 1
            col = (idx % number_of_columns) + 1

            figure = _plot_stacked_bar(
                figure=figure,
                row=row,
                col=col,
                chunker=chunker,
                column_name=d1_value,
                metric_display_name=d2_value.display_name,
                reference_data=reference_data[d1_value],
                reference_data_timestamps=reference_data[result.timestamp_column_name],
                reference_alerts=reference_result.get((d1_value, d2_value.column_name, 'alert'), default=None),
                reference_chunk_keys=reference_result.get(('chunk', 'chunk', 'key'), default=None),
                reference_chunk_periods=reference_result.get(('chunk', 'chunk', 'period'), default=None),
                reference_chunk_indices=reference_result.get(('chunk', 'chunk', 'chunk_index'), default=None),
                reference_chunk_start_dates=reference_result.get(('chunk', 'chunk', 'start_date'), default=None),
                reference_chunk_end_dates=reference_result.get(('chunk', 'chunk', 'end_date'), default=None),
                analysis_data=analysis_data[d1_value],
                analysis_data_timestamps=analysis_data[result.timestamp_column_name],
                analysis_alerts=analysis_result.get((d1_value, d2_value.column_name, 'alert'), default=None),
                analysis_chunk_keys=analysis_result.get(('chunk', 'chunk', 'key'), default=None),
                analysis_chunk_periods=analysis_result.get(('chunk', 'chunk', 'period'), default=None),
                analysis_chunk_indices=analysis_result.get(('chunk', 'chunk', 'chunk_index'), default=None),
                analysis_chunk_start_dates=analysis_result.get(('chunk', 'chunk', 'start_date'), default=None),
                analysis_chunk_end_dates=analysis_result.get(('chunk', 'chunk', 'end_date'), default=None),
            )

    return figure


def _try_get_dimension(result, dimension_name: str) -> List:
    try:
        dimension = getattr(result, dimension_name)
    except AttributeError as exc:
        raise InvalidArgumentsException(f'result does not contain an attribute named {dimension_name}: {exc}')

    if not isinstance(dimension, List):
        raise InvalidArgumentsException(f'attribute {dimension_name} is not an instance of \'List\'')

    return dimension


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
