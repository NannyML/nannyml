#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import copy
import math
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from nannyml._typing import Key, Result
from nannyml.plots import Colors, ensure_numpy, has_non_null_data, is_time_based_x_axis
from nannyml.plots.components import Figure, Hover, render_alert_string, render_period_string, render_x_coordinate


def plot_metrics(
    result: Result,
    title: str,
    x_axis_time_title: str = 'Time',
    x_axis_chunk_title: str = 'Chunk',
    y_axis_title: str = 'Metric',
    figure_args: Optional[Dict[str, Any]] = None,
    subplot_title_format: str = '<b>{display_names}</b>',
    subplot_y_axis_title_format: str = '<b>{display_names}</b>',
    number_of_columns: Optional[int] = None,
    hover: Optional[Hover] = None,
    **kwargs,
) -> Figure:
    if number_of_columns is None:
        number_of_columns = min(len(result.keys()), 1)

    reference_result = result.filter(period='reference')
    analysis_result = result.filter(period='analysis')

    if figure_args is None:
        figure_args = {}

    figure = Figure(
        **dict(
            title=title,
            x_axis_title=x_axis_time_title
            if is_time_based_x_axis(result.chunk_start_dates, result.chunk_end_dates)
            else x_axis_chunk_title,
            y_axis_title=y_axis_title,
            legend=dict(traceorder="grouped", itemclick=False, itemdoubleclick=False),
            height=len(result.keys()) * 500 / number_of_columns,
            subplot_args=dict(
                cols=number_of_columns,
                rows=math.ceil(len(result.keys()) / number_of_columns),
                subplot_titles=[subplot_title_format.format(display_names=key.display_names) for key in result.keys()],
            ),
            **figure_args,
        )
    )

    for idx, key in enumerate(result.keys()):
        row = (idx // number_of_columns) + 1
        col = (idx % number_of_columns) + 1

        figure = _plot_metric(
            figure=figure,
            row=row,
            col=col,
            metric_display_name=subplot_title_format.format(display_names=key.display_names),
            reference_metric=reference_result.values(key),
            reference_alerts=reference_result.alerts(key),
            reference_chunk_keys=reference_result.chunk_keys,
            reference_chunk_periods=reference_result.chunk_periods,
            reference_chunk_indices=reference_result.chunk_indices,
            reference_chunk_start_dates=reference_result.chunk_start_dates,
            reference_chunk_end_dates=reference_result.chunk_end_dates,
            reference_upper_thresholds=reference_result.upper_thresholds(key),
            reference_lower_thresholds=reference_result.lower_thresholds(key),
            reference_upper_confidence_boundary=reference_result.upper_confidence_bounds(key),
            reference_lower_confidence_boundary=reference_result.lower_confidence_bounds(key),
            reference_sampling_error=reference_result.sampling_error(key),
            analysis_metric=analysis_result.values(key),
            analysis_alerts=analysis_result.alerts(key),
            analysis_chunk_keys=analysis_result.chunk_keys,
            analysis_chunk_periods=analysis_result.chunk_periods,
            analysis_chunk_indices=analysis_result.chunk_indices,
            analysis_chunk_start_dates=analysis_result.chunk_start_dates,
            analysis_chunk_end_dates=analysis_result.chunk_end_dates,
            analysis_upper_thresholds=analysis_result.upper_thresholds(key),
            analysis_lower_thresholds=analysis_result.lower_thresholds(key),
            analysis_upper_confidence_boundary=analysis_result.upper_confidence_bounds(key),
            analysis_lower_confidence_boundary=analysis_result.lower_confidence_bounds(key),
            analysis_sampling_error=analysis_result.sampling_error(key),
            hover=hover,
            subplot_y_axis_title=subplot_y_axis_title_format.format(display_names=key.display_names),
            **kwargs,
        )

    return figure


def plot_metric(
    result: Result,
    title: str,
    metric_display_name: str,
    metric_column_name: str,
    x_axis_time_title: str = 'Time',
    x_axis_chunk_title: str = 'Chunk',
    y_axis_title: str = 'Metric',
    figure_args: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Figure:
    reference_result = result.filter(period='reference')
    analysis_result = result.filter(period='analysis')

    if figure_args is None:
        figure_args = {}

    figure = Figure(
        **dict(
            title=title,
            x_axis_title=x_axis_time_title
            if is_time_based_x_axis(reference_result.chunk_start_dates, reference_result.chunk_end_dates)
            else x_axis_chunk_title,
            y_axis_title=y_axis_title,
            legend=dict(traceorder="grouped", itemclick=False, itemdoubleclick=False),
            **figure_args,
        )
    )

    key = Key(properties=(metric_column_name,), display_names=(metric_display_name,))

    figure = _plot_metric(
        figure=figure,
        metric_display_name=metric_display_name,
        reference_metric=reference_result.values(key),
        reference_alerts=reference_result.alerts(key),
        reference_chunk_keys=reference_result.chunk_keys,
        reference_chunk_periods=reference_result.chunk_periods,
        reference_chunk_indices=reference_result.chunk_indices,
        reference_chunk_start_dates=reference_result.chunk_start_dates,
        reference_chunk_end_dates=reference_result.chunk_end_dates,
        reference_upper_thresholds=reference_result.upper_thresholds(key),
        reference_lower_thresholds=reference_result.lower_thresholds(key),
        reference_upper_confidence_boundary=reference_result.upper_confidence_bounds(key),
        reference_lower_confidence_boundary=reference_result.lower_confidence_bounds(key),
        reference_sampling_error=reference_result.sampling_error(key),
        analysis_metric=analysis_result.values(key),
        analysis_alerts=analysis_result.alerts(key),
        analysis_chunk_keys=analysis_result.chunk_keys,
        analysis_chunk_periods=analysis_result.chunk_periods,
        analysis_chunk_indices=analysis_result.chunk_indices,
        analysis_chunk_start_dates=analysis_result.chunk_start_dates,
        analysis_chunk_end_dates=analysis_result.chunk_end_dates,
        analysis_upper_thresholds=analysis_result.upper_thresholds(key),
        analysis_lower_thresholds=analysis_result.lower_thresholds(key),
        analysis_upper_confidence_boundary=analysis_result.upper_confidence_bounds(key),
        analysis_lower_confidence_boundary=analysis_result.lower_confidence_bounds(key),
        analysis_sampling_error=analysis_result.sampling_error(key),
        **kwargs,
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
    subplot_y_axis_title: Optional[str] = None,
    color: Optional[str] = None,
    metric_name: str = 'Metric',
    **kwargs,
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
            _hover.add(render_period_string(reference_chunk_periods, color), name='period')

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
            name=metric_name,
            color=color or Colors.BLUE_SKY_CRAYOLA,
            hover=_hover,
            # subplot_args=dict(row=row, col=col, subplot_y_axis_title=metric_display_name),
            subplot_args=dict(row=row, col=col, subplot_y_axis_title=subplot_y_axis_title or metric_display_name),
            legendgroup='metric_reference',
            showlegend=show_in_legend,
            # line_dash='dash',
            **kwargs,
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
        _hover.add(render_period_string(analysis_chunk_periods, color), name='period')

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
        name=metric_name,
        color=color or Colors.BLUE_SKY_CRAYOLA,
        hover=_hover,
        subplot_args=dict(row=row, col=col, subplot_y_axis_title=subplot_y_axis_title or metric_display_name),
        legendgroup='metric_analysis',
        showlegend=show_in_legend and not has_reference_results,
        **kwargs,
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
    show_threshold_legend = show_in_legend
    if has_reference_results:
        if has_non_null_data(reference_upper_thresholds):
            figure.add_threshold(
                data=reference_upper_thresholds,
                indices=reference_chunk_indices,
                start_dates=reference_chunk_start_dates,
                end_dates=reference_chunk_end_dates,
                name='Threshold',
                with_additional_endpoint=True,
                subplot_args=dict(row=row, col=col),
                legendgroup='thresh',
                showlegend=show_threshold_legend,
            )
            show_threshold_legend = False
        if has_non_null_data(reference_lower_thresholds):
            figure.add_threshold(
                data=reference_lower_thresholds,
                indices=reference_chunk_indices,
                start_dates=reference_chunk_start_dates,
                end_dates=reference_chunk_end_dates,
                name='Threshold',
                with_additional_endpoint=True,
                subplot_args=dict(row=row, col=col),
                legendgroup='thresh',
                showlegend=show_threshold_legend,
            )
            show_threshold_legend = False
    if has_non_null_data(analysis_upper_thresholds):
        figure.add_threshold(
            data=analysis_upper_thresholds,
            indices=analysis_chunk_indices,
            start_dates=analysis_chunk_start_dates,
            end_dates=analysis_chunk_end_dates,
            name='Threshold',
            with_additional_endpoint=True,
            subplot_args=dict(row=row, col=col),
            legendgroup='thresh',
            showlegend=show_threshold_legend and not has_reference_results,
        )
        show_threshold_legend = False

    if has_non_null_data(analysis_lower_thresholds):
        figure.add_threshold(
            data=analysis_lower_thresholds,
            indices=analysis_chunk_indices,
            start_dates=analysis_chunk_start_dates,
            end_dates=analysis_chunk_end_dates,
            name='Threshold',
            with_additional_endpoint=True,
            subplot_args=dict(row=row, col=col),
            legendgroup='thresh',
            showlegend=show_threshold_legend and not has_reference_results,
        )
        show_threshold_legend = False
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
                name='Confidence band',
                color=color or Colors.BLUE_SKY_CRAYOLA,
                with_additional_endpoint=True,
                subplot_args=dict(row=row, col=col),
                showlegend=show_in_legend,
            )

    if analysis_upper_confidence_boundary is not None and analysis_lower_confidence_boundary is not None:
        figure.add_confidence_band(
            upper_confidence_boundaries=analysis_upper_confidence_boundary,
            lower_confidence_boundaries=analysis_lower_confidence_boundary,
            indices=analysis_chunk_indices,
            start_dates=analysis_chunk_start_dates,
            end_dates=analysis_chunk_end_dates,
            name='Confidence band',
            color=color or Colors.BLUE_SKY_CRAYOLA,
            with_additional_endpoint=True,
            subplot_args=dict(row=row, col=col),
            showlegend=show_in_legend and not has_reference_results,
        )
    # endregion

    # region Period separator
    if has_reference_results:
        is_time_based = is_time_based_x_axis(reference_chunk_start_dates, reference_chunk_end_dates)

        figure.add_period_separator(
            x=(
                ensure_numpy(reference_chunk_indices)[0][-1] + 1
                if not is_time_based
                else ensure_numpy(analysis_chunk_start_dates)[0][0]
            )
        )

        reference_period_text_x = (
            reference_chunk_indices.mean() if not is_time_based else reference_chunk_start_dates.mean()  # type: ignore
        )

        figure.add_annotation(
            x=reference_period_text_x,
            xshift=10,
            yref='y domain',
            y=1.01,
            text="Reference",
            showarrow=False,
            row=row,
            col=col,
        )

        analyis_period_text_x = (
            analysis_chunk_indices.mean() if not is_time_based else analysis_chunk_start_dates.mean()  # type: ignore
        )

        figure.add_annotation(
            x=analyis_period_text_x,
            xshift=15,
            yref='y domain',
            y=1.01,
            text="Analysis",
            showarrow=False,
            row=row,
            col=col,
        )
    # endregion

    return figure
