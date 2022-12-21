#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import copy
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from nannyml.exceptions import InvalidArgumentsException
from nannyml.plots import Colors, ensure_numpy, is_time_based_x_axis
from nannyml.plots.components import Figure, Hover, render_alert_string, render_period_string, render_x_coordinate


def plot_2d_metric_list(
    result,
    title: str,
    items: List[Tuple[str, Any]],
    x_axis_time_title: str = 'Time',
    x_axis_chunk_title: str = 'Chunk',
    y_axis_title: str = 'Metric',
    figure_args: Optional[Dict[str, Any]] = None,
    subplot_title_format: str = '{column_name} (<b>{method_name}</b>)',
    hover: Optional[Hover] = None,
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
                    subplot_title_format.format(column_name=column_name, method_name=method_name.display_name)
                    for column_name, method_name in items
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

        figure = _plot_metric(
            figure=figure,
            row=row,
            col=col,
            metric_display_name=method.display_name,
            reference_metric=reference_result[(column_name, method.column_name, 'value')],
            reference_alerts=reference_result.get((column_name, method.column_name, 'alert'), default=None),
            reference_chunk_keys=reference_result.get(('chunk', 'chunk', 'key'), default=None),
            reference_chunk_periods=reference_result.get(('chunk', 'chunk', 'period'), default=None),
            reference_chunk_indices=reference_result.get(('chunk', 'chunk', 'chunk_index'), default=None),
            reference_chunk_start_dates=reference_result.get(('chunk', 'chunk', 'start_date'), default=None),
            reference_chunk_end_dates=reference_result.get(('chunk', 'chunk', 'end_date'), default=None),
            reference_upper_thresholds=reference_result.get(
                (column_name, method.column_name, 'upper_threshold'), default=None
            ),
            reference_lower_thresholds=reference_result.get(
                (column_name, method.column_name, 'lower_threshold'), default=None
            ),
            reference_upper_confidence_boundary=reference_result.get(
                (column_name, method.column_name, 'upper_confidence_boundary'), default=None
            ),
            reference_lower_confidence_boundary=reference_result.get(
                (column_name, method.column_name, 'lower_confidence_boundary'), default=None
            ),
            reference_sampling_error=reference_result.get(
                (column_name, method.column_name, 'sampling_error'), default=None
            ),
            analysis_metric=analysis_result[(column_name, method.column_name, 'value')],
            analysis_alerts=analysis_result.get((column_name, method.column_name, 'alert'), default=None),
            analysis_chunk_keys=analysis_result.get(('chunk', 'chunk', 'key'), default=None),
            analysis_chunk_periods=analysis_result.get(('chunk', 'chunk', 'period'), default=None),
            analysis_chunk_indices=analysis_result.get(('chunk', 'chunk', 'chunk_index'), default=None),
            analysis_chunk_start_dates=analysis_result.get(('chunk', 'chunk', 'start_date'), default=None),
            analysis_chunk_end_dates=analysis_result.get(('chunk', 'chunk', 'end_date'), default=None),
            analysis_upper_thresholds=analysis_result.get(
                (column_name, method.column_name, 'upper_threshold'), default=None
            ),
            analysis_lower_thresholds=analysis_result.get(
                (column_name, method.column_name, 'lower_threshold'), default=None
            ),
            analysis_upper_confidence_boundary=analysis_result.get(
                (column_name, method.column_name, 'upper_confidence_boundary'), default=None
            ),
            analysis_lower_confidence_boundary=analysis_result.get(
                (column_name, method.column_name, 'lower_confidence_boundary'), default=None
            ),
            analysis_sampling_error=analysis_result.get(
                (column_name, method.column_name, 'sampling_error'), default=None
            ),
            hover=hover,
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


def plot_metric_list(
    result,
    title: str,
    x_axis_time_title: str = 'Time',
    x_axis_chunk_title: str = 'Chunk',
    y_axis_title: str = 'Metric',
    figure_args: Optional[Dict[str, Any]] = None,
    subplot_title_format: str = '<b>{metric_name}</b>',
    number_of_columns: Optional[int] = None,
) -> Figure:
    if number_of_columns is None:
        number_of_columns = min(len(result.metrics), 1)

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

        figure = _plot_metric(
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
            reference_upper_thresholds=reference_result.get((metric.column_name, 'upper_threshold'), default=None),
            reference_lower_thresholds=reference_result.get((metric.column_name, 'lower_threshold'), default=None),
            reference_upper_confidence_boundary=reference_result.get(
                (metric.column_name, 'upper_confidence_boundary'), default=None
            ),
            reference_lower_confidence_boundary=reference_result.get(
                (metric.column_name, 'lower_confidence_boundary'), default=None
            ),
            reference_sampling_error=reference_result.get((metric.column_name, 'sampling_error'), default=None),
            analysis_metric=analysis_result[(metric.column_name, 'value')],
            analysis_alerts=analysis_result.get((metric.column_name, 'alert'), default=None),
            analysis_chunk_keys=analysis_result.get(('chunk', 'key'), default=None),
            analysis_chunk_periods=analysis_result.get(('chunk', 'period'), default=None),
            analysis_chunk_indices=analysis_result.get(('chunk', 'chunk_index'), default=None),
            analysis_chunk_start_dates=analysis_result.get(('chunk', 'start_date'), default=None),
            analysis_chunk_end_dates=analysis_result.get(('chunk', 'end_date'), default=None),
            analysis_upper_thresholds=analysis_result.get((metric.column_name, 'upper_threshold'), default=None),
            analysis_lower_thresholds=analysis_result.get((metric.column_name, 'lower_threshold'), default=None),
            analysis_upper_confidence_boundary=analysis_result.get(
                (metric.column_name, 'upper_confidence_boundary'), default=None
            ),
            analysis_lower_confidence_boundary=analysis_result.get(
                (metric.column_name, 'lower_confidence_boundary'), default=None
            ),
            analysis_sampling_error=analysis_result.get((metric.column_name, 'sampling_error'), default=None),
        )

    return figure


def plot_metric(
    result,
    title: str,
    metric_display_name: str,
    metric_column_name: str,
    x_axis_time_title: str = 'Time',
    x_axis_chunk_title: str = 'Chunk',
    y_axis_title: str = 'Metric',
    figure_args: Optional[Dict[str, Any]] = None,
) -> Figure:
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
            **figure_args,
        )
    )

    figure = _plot_metric(
        figure=figure,
        metric_display_name=metric_display_name,
        reference_metric=reference_result[(metric_column_name, 'value')],
        reference_alerts=reference_result.get((metric_column_name, 'alert'), default=None),
        reference_chunk_keys=reference_result.get(('chunk', 'key'), default=None),
        reference_chunk_periods=reference_result.get(('chunk', 'period'), default=None),
        reference_chunk_indices=reference_result.get(('chunk', 'chunk_index'), default=None),
        reference_chunk_start_dates=reference_result.get(('chunk', 'start_date'), default=None),
        reference_chunk_end_dates=reference_result.get(('chunk', 'end_date'), default=None),
        reference_upper_thresholds=reference_result.get((metric_column_name, 'upper_threshold'), default=None),
        reference_lower_thresholds=reference_result.get((metric_column_name, 'lower_threshold'), default=None),
        reference_upper_confidence_boundary=reference_result.get(
            ('reconstruction_error', 'upper_confidence_boundary'), default=None
        ),
        reference_lower_confidence_boundary=reference_result.get(
            ('reconstruction_error', 'lower_confidence_boundary'), default=None
        ),
        reference_sampling_error=reference_result.get((metric_column_name, 'sampling_error'), default=None),
        analysis_metric=analysis_result[(metric_column_name, 'value')],
        analysis_alerts=analysis_result.get((metric_column_name, 'alert'), default=None),
        analysis_chunk_keys=analysis_result.get(('chunk', 'key'), default=None),
        analysis_chunk_periods=analysis_result.get(('chunk', 'period'), default=None),
        analysis_chunk_indices=analysis_result.get(('chunk', 'chunk_index'), default=None),
        analysis_chunk_start_dates=analysis_result.get(('chunk', 'start_date'), default=None),
        analysis_chunk_end_dates=analysis_result.get(('chunk', 'end_date'), default=None),
        analysis_upper_thresholds=analysis_result.get((metric_column_name, 'upper_threshold'), default=None),
        analysis_lower_thresholds=analysis_result.get((metric_column_name, 'lower_threshold'), default=None),
        analysis_upper_confidence_boundary=analysis_result.get(
            (metric_column_name, 'upper_confidence_boundary'), default=None
        ),
        analysis_lower_confidence_boundary=analysis_result.get(
            (metric_column_name, 'lower_confidence_boundary'), default=None
        ),
        analysis_sampling_error=analysis_result.get((metric_column_name, 'sampling_error'), default=None),
    )

    return figure


def _plot_metric(  # noqa: C901
    figure: Figure,
    metric_display_name: str,
    analysis_metric: Union[np.ndarray, pd.Series],
    reference_metric: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_alerts: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_keys: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_periods: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_indices: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_start_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_end_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_upper_thresholds: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_lower_thresholds: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_upper_confidence_boundary: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_lower_confidence_boundary: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_sampling_error: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_alerts: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_chunk_keys: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_chunk_periods: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_chunk_indices: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_chunk_start_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_chunk_end_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_upper_thresholds: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_lower_thresholds: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_upper_confidence_boundary: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_lower_confidence_boundary: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_sampling_error: Optional[Union[np.ndarray, pd.Series]] = None,
    row: Optional[int] = None,
    col: Optional[int] = None,
    hover: Optional[Hover] = None,
) -> Figure:

    if figure is None:
        figure = Figure(
            title='CBPE',
            x_axis_title='Time'
            if is_time_based_x_axis(reference_chunk_start_dates, reference_chunk_end_dates)
            else 'Chunk',
            y_axis_title='Estimated metric',
            legend=dict(traceorder="grouped", itemclick=False, itemdoubleclick=False),
            height=500,
        )

    is_subplot = row is not None and col is not None

    show_in_legend = row == 1 and col == 1 if is_subplot else True

    has_reference_results = reference_chunk_indices is not None and len(reference_chunk_indices) > 0

    # region reference metric

    if has_reference_results:
        assert reference_metric is not None
        if hover is None:
            _hover = Hover(
                template='%{period} &nbsp; &nbsp; %{alert} <br />'
                'Chunk: <b>%{chunk_key}</b> &nbsp; &nbsp; %{x_coordinate} <br />'
                '%{metric_name}: <b>%{metric_value}</b><br />'
                'Confidence band: +/- <b>%{sampling_error}</b><br />',
                show_extra=True,
            )
        else:
            _hover = copy.deepcopy(hover)
        _hover.add(np.asarray([metric_display_name] * len(reference_metric)), 'metric_name')

        if reference_chunk_periods is not None:
            _hover.add(render_period_string(reference_chunk_periods), name='period')

        if reference_alerts is not None:
            _hover.add(render_alert_string(reference_alerts), name='alert')

        if reference_chunk_keys is not None:
            _hover.add(reference_chunk_keys, name='chunk_key')

        _hover.add(
            render_x_coordinate(
                reference_chunk_indices,
                reference_chunk_start_dates,
                reference_chunk_end_dates,
            ),
            name='x_coordinate',
        )

        _hover.add(np.round(reference_metric, 4), name='metric_value')

        if reference_sampling_error is not None:
            _hover.add(np.round(reference_sampling_error * 3, 4), name='sampling_error')

        figure.add_metric(
            data=reference_metric,
            indices=reference_chunk_indices,
            start_dates=reference_chunk_start_dates,
            end_dates=reference_chunk_end_dates,
            name='Metric (reference)',
            color=Colors.BLUE_SKY_CRAYOLA,
            hover=_hover,
            subplot_args=dict(row=row, col=col, subplot_y_axis_title=metric_display_name),
            legendgroup='metric_reference',
            showlegend=show_in_legend,
            # line_dash='dash',
        )
    # endregion

    # region analysis metrics
    if hover is None:
        _hover = Hover(
            template='%{period} &nbsp; &nbsp; %{alert} <br />'
            'Chunk: <b>%{chunk_key}</b> &nbsp; &nbsp; %{x_coordinate} <br />'
            '%{metric_name}: <b>%{metric_value}</b><br />'
            'Confidence band: +/- <b>%{sampling_error}</b><br />',
            show_extra=True,
        )
    else:
        _hover = copy.deepcopy(hover)

    _hover.add(np.asarray([metric_display_name] * len(analysis_metric)), 'metric_name')

    if analysis_chunk_periods is not None:
        _hover.add(render_period_string(analysis_chunk_periods), name='period')

    if analysis_alerts is not None:
        _hover.add(render_alert_string(analysis_alerts), name='alert')

    if analysis_chunk_keys is not None:
        _hover.add(analysis_chunk_keys, name='chunk_key')

    _hover.add(
        render_x_coordinate(
            analysis_chunk_indices,
            analysis_chunk_start_dates,
            analysis_chunk_end_dates,
        ),
        name='x_coordinate',
    )
    _hover.add(np.round(analysis_metric, 4), name='metric_value')

    if analysis_sampling_error is not None:
        _hover.add(np.round(analysis_sampling_error * 3, 4), name='sampling_error')

    if has_reference_results:
        assert reference_chunk_indices is not None
        analysis_chunk_indices = analysis_chunk_indices + (max(reference_chunk_indices) + 1)

    figure.add_metric(
        data=analysis_metric,
        indices=analysis_chunk_indices,
        start_dates=analysis_chunk_start_dates,
        end_dates=analysis_chunk_end_dates,
        name='Metric (analysis)',
        color=Colors.INDIGO_PERSIAN,
        hover=_hover,
        subplot_args=dict(row=row, col=col, subplot_y_axis_title=metric_display_name),
        legendgroup='metric_analysis',
        showlegend=show_in_legend,
        # line_dash='dash',
    )
    # endregion

    # region alert

    if analysis_alerts is not None:
        figure.add_alert(
            data=analysis_metric,
            alerts=analysis_alerts,
            indices=analysis_chunk_indices,
            start_dates=analysis_chunk_start_dates,
            end_dates=analysis_chunk_end_dates,
            name='Alert',
            subplot_args=dict(row=row, col=col),
            legendgroup='alert',
            plot_areas=False,
            showlegend=show_in_legend,
        )

    # endregion

    # region thresholds

    if has_reference_results:
        if reference_upper_thresholds is not None:
            figure.add_threshold(
                data=reference_upper_thresholds,
                indices=reference_chunk_indices,
                start_dates=reference_chunk_start_dates,
                end_dates=reference_chunk_end_dates,
                name='Threshold',
                with_additional_endpoint=True,
                subplot_args=dict(row=row, col=col),
                legendgroup='thresh',
                showlegend=False,
            )
        if reference_lower_thresholds is not None:
            figure.add_threshold(
                data=reference_lower_thresholds,
                indices=reference_chunk_indices,
                start_dates=reference_chunk_start_dates,
                end_dates=reference_chunk_end_dates,
                name='Threshold',
                with_additional_endpoint=True,
                subplot_args=dict(row=row, col=col),
                legendgroup='thresh',
                showlegend=True,
            )
    if analysis_upper_thresholds is not None:
        figure.add_threshold(
            data=analysis_upper_thresholds,
            indices=analysis_chunk_indices,
            start_dates=analysis_chunk_start_dates,
            end_dates=analysis_chunk_end_dates,
            name='Threshold',
            with_additional_endpoint=True,
            subplot_args=dict(row=row, col=col),
            legendgroup='thresh',
            showlegend=False,
        )

    if analysis_lower_thresholds is not None:
        figure.add_threshold(
            data=analysis_lower_thresholds,
            indices=analysis_chunk_indices,
            start_dates=analysis_chunk_start_dates,
            end_dates=analysis_chunk_end_dates,
            name='Threshold',
            with_additional_endpoint=True,
            subplot_args=dict(row=row, col=col),
            legendgroup='thresh',
            showlegend=False,
        )
    # endregion

    # region confidence bands

    if has_reference_results:
        if reference_upper_confidence_boundary is not None and reference_lower_confidence_boundary is not None:
            figure.add_confidence_band(
                upper_confidence_boundaries=reference_upper_confidence_boundary,
                lower_confidence_boundaries=reference_lower_confidence_boundary,
                indices=reference_chunk_indices,
                start_dates=reference_chunk_start_dates,
                end_dates=reference_chunk_end_dates,
                name='Sampling error (reference)',
                color=Colors.BLUE_SKY_CRAYOLA,
                with_additional_endpoint=True,
                subplot_args=dict(row=row, col=col),
                showlegend=show_in_legend,
            )

    if analysis_upper_confidence_boundary is not None and analysis_upper_confidence_boundary is not None:
        figure.add_confidence_band(
            upper_confidence_boundaries=analysis_upper_confidence_boundary,
            lower_confidence_boundaries=analysis_lower_confidence_boundary,
            indices=analysis_chunk_indices,
            start_dates=analysis_chunk_start_dates,
            end_dates=analysis_chunk_end_dates,
            name='Sampling error (analysis)',
            color=Colors.INDIGO_PERSIAN,
            with_additional_endpoint=True,
            subplot_args=dict(row=row, col=col),
            showlegend=show_in_legend,
        )
    # endregion

    if has_reference_results:
        assert reference_chunk_indices is not None
        reference_chunk_indices, analysis_chunk_start_dates = ensure_numpy(
            reference_chunk_indices, analysis_chunk_start_dates
        )
        figure.add_period_separator(
            x=(
                reference_chunk_indices[-1] + 1
                if not is_time_based_x_axis(reference_chunk_start_dates, reference_chunk_end_dates)
                else analysis_chunk_start_dates[0]  # type: ignore
            ),
            subplot_args=dict(row=row, col=col),
        )

    return figure
