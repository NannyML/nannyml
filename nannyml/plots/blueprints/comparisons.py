#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from nannyml._typing import Metric, Result
from nannyml.base import AbstractCalculatorResult, AbstractEstimatorResult
from nannyml.drift.multivariate.data_reconstruction import Result as DataReconstructionDriftResult
from nannyml.exceptions import InvalidArgumentsException
from nannyml.plots import Colors
from nannyml.plots.components import Figure, Hover, render_alert_string, render_period_string, render_x_coordinate
from nannyml.plots.util import ensure_numpy, is_time_based_x_axis


def plot_compare_performance_to_drift(
    performance_result: Union[AbstractCalculatorResult, AbstractEstimatorResult],
    drift_result: Union[DataReconstructionDriftResult],
    performance_metric_display_name: str,
    drift_metric_display_name: str,
    performance_metric_column_name: str,
    drift_metric_column_name: str,
) -> Figure:
    reference_performance_result = performance_result.filter(period='reference').to_df()
    analysis_performance_result = performance_result.filter(period='analysis').to_df()

    reference_drift_result = drift_result.filter(period='reference').to_df()
    analysis_drift_result = drift_result.filter(period='analysis').to_df()

    reference_chunk_indices = reference_performance_result[('chunk', 'chunk_index')]
    reference_chunk_start_dates = reference_performance_result[('chunk', 'start_date')]
    reference_chunk_end_dates = reference_performance_result[('chunk', 'end_date')]
    reference_chunk_periods = reference_performance_result[('chunk', 'period')]
    reference_chunk_keys = reference_performance_result[('chunk', 'key')]
    reference_performance_metric = reference_performance_result[(performance_metric_column_name, 'value')]
    reference_drift_metric = reference_drift_result[('reconstruction_error', 'value')]

    analysis_chunk_indices = analysis_performance_result[('chunk', 'chunk_index')]
    analysis_chunk_start_dates = analysis_performance_result[('chunk', 'start_date')]
    analysis_chunk_end_dates = analysis_performance_result[('chunk', 'end_date')]
    analysis_chunk_periods = analysis_performance_result[('chunk', 'period')]
    analysis_chunk_keys = analysis_performance_result[('chunk', 'key')]
    analysis_performance_metric = analysis_performance_result[(performance_metric_column_name, 'value')]
    analysis_drift_metric = analysis_drift_result[(drift_metric_column_name, 'value')]

    analysis_performance_alerts = analysis_performance_result[(performance_metric_column_name, 'alert')]
    analysis_drift_alerts = analysis_drift_result[(drift_metric_column_name, 'alert')]

    figure = Figure(
        title='CBPE',
        x_axis_title='Time'
        if is_time_based_x_axis(reference_chunk_start_dates, reference_chunk_end_dates)
        else 'Chunk',
        y_axis_title='Estimated metric',
        legend=dict(traceorder="grouped", itemclick=False, itemdoubleclick=False),
        height=500,
        yaxis2=dict(
            title="Reconstruction error",
            anchor="x",
            overlaying="y",
            side="right",
        ),
    )

    return _plot_compare_step_to_step(
        figure=figure,
        metric_1_display_name=performance_metric_display_name,
        metric_2_display_name=drift_metric_display_name,
        analysis_metric_1=analysis_performance_metric,
        analysis_metric_2=analysis_drift_metric,
        reference_chunk_keys=reference_chunk_keys,
        reference_chunk_periods=reference_chunk_periods,
        reference_chunk_indices=reference_chunk_indices,
        reference_chunk_start_dates=reference_chunk_start_dates,
        reference_chunk_end_dates=reference_chunk_end_dates,
        reference_metric_1=reference_performance_metric,
        reference_metric_2=reference_drift_metric,
        analysis_chunk_keys=analysis_chunk_keys,
        analysis_chunk_periods=analysis_chunk_periods,
        analysis_chunk_indices=analysis_chunk_indices,
        analysis_chunk_start_dates=analysis_chunk_start_dates,
        analysis_chunk_end_dates=analysis_chunk_end_dates,
        analysis_metric_1_alerts=analysis_performance_alerts,
        analysis_metric_2_alerts=analysis_drift_alerts,
        hover=None,
    )


def plot_2d_compare_step_to_step(
    result_1: Result,
    result_2: Result,
    items: List[Tuple[Metric, Metric]],
    plot_title: str,
    x_axis_time_title: str = 'Time',
    x_axis_chunk_title: str = 'Chunk',
    y_axis_title: str = 'Comparison',
    subplot_titles: Optional[List[str]] = None,
    number_of_columns: Optional[int] = None,
    hover: Optional[Hover] = None,
) -> Figure:
    if len(items) == 0:
        raise InvalidArgumentsException("tried plotting comparisons but received zero plotting items.")

    number_of_plots = len(items)
    if number_of_columns is None:
        number_of_columns = min(number_of_plots, 1)
    number_of_rows = math.ceil(number_of_plots / number_of_columns)

    reference_result_1: pd.DataFrame = result_1.filter(period='reference').to_df()
    analysis_result_1: pd.DataFrame = result_1.filter(period='analysis').to_df()
    reference_result_2: pd.DataFrame = result_2.filter(period='reference').to_df()
    analysis_result_2: pd.DataFrame = result_2.filter(period='analysis').to_df()

    reference_chunk_indices = reference_result_1[('chunk', 'chunk_index')]
    reference_chunk_start_dates = reference_result_1[('chunk', 'start_date')]
    reference_chunk_end_dates = reference_result_1[('chunk', 'end_date')]
    reference_chunk_periods = reference_result_1[('chunk', 'period')]
    reference_chunk_keys = reference_result_1[('chunk', 'key')]

    analysis_chunk_indices = analysis_result_1[('chunk', 'chunk_index')]
    analysis_chunk_start_dates = analysis_result_1[('chunk', 'start_date')]
    analysis_chunk_end_dates = analysis_result_1[('chunk', 'end_date')]
    analysis_chunk_periods = analysis_result_1[('chunk', 'period')]
    analysis_chunk_keys = analysis_result_1[('chunk', 'key')]

    # region setup axes

    x_axis_title = (
        x_axis_time_title
        if is_time_based_x_axis(reference_chunk_start_dates, reference_chunk_end_dates)
        else x_axis_chunk_title
    )
    figure = Figure(
        plot_title,
        x_axis_title,
        y_axis_title,
        legend=dict(traceorder="grouped", itemclick=False, itemdoubleclick=False),
        height=number_of_plots * 500 / number_of_columns,
    )

    subplot_specs = [[{"secondary_y": True} for _ in range(number_of_columns)] for _ in range(number_of_rows)]
    if subplot_titles is None:
        subplot_titles = [f'{metric_1.display_name} vs. {metric_2.display_name}' for metric_1, metric_2 in items]
    figure.set_subplots(
        rows=number_of_rows,
        cols=number_of_columns,
        specs=subplot_specs,
        subplot_titles=subplot_titles,
    )
    figure.update_xaxes(
        title=x_axis_title,
        linecolor=Colors.INDIGO_PERSIAN,
        showgrid=False,
        showticklabels=is_time_based_x_axis(analysis_chunk_start_dates, analysis_chunk_end_dates),
        mirror=True,
        zeroline=False,
        matches='x',
        row=None,
        col=None,
    )
    figure.update_yaxes(
        title=y_axis_title,
        linecolor=Colors.INDIGO_PERSIAN,
        showgrid=False,
        range=figure.layout.yaxis.range,
        mirror=True,
        zeroline=False,
        row=None,
        col=None,
    )

    # endregion

    for idx, (metric_1, metric_2) in enumerate(items):
        reference_metric_1 = reference_result_1[(metric_1.column_name, 'value')]
        reference_metric_2 = reference_result_2[(metric_2.column_name, 'value')]
        analysis_metric_1 = analysis_result_1[(metric_1.column_name, 'value')]
        analysis_metric_2 = analysis_result_2[(metric_2.column_name, 'value')]
        analysis_metric_1_alerts = analysis_result_1[(metric_1.column_name, 'alert')]
        analysis_metric_2_alerts = analysis_result_2[(metric_2.column_name, 'alert')]

        x_axis, y_axis, y_axis_2 = _get_subplot_axes_names(idx, y_axis_per_subplot=2)

        figure = _plot_compare_step_to_step(
            figure=figure,
            metric_1_display_name=metric_1.display_name,
            metric_2_display_name=metric_2.display_name,
            analysis_metric_1=analysis_metric_1,
            analysis_metric_2=analysis_metric_2,
            reference_chunk_keys=reference_chunk_keys,
            reference_chunk_periods=reference_chunk_periods,
            reference_chunk_indices=reference_chunk_indices,
            reference_chunk_start_dates=reference_chunk_start_dates,
            reference_chunk_end_dates=reference_chunk_end_dates,
            reference_metric_1=reference_metric_1,
            reference_metric_2=reference_metric_2,
            analysis_chunk_keys=analysis_chunk_keys,
            analysis_chunk_periods=analysis_chunk_periods,
            analysis_chunk_indices=analysis_chunk_indices,
            analysis_chunk_start_dates=analysis_chunk_start_dates,
            analysis_chunk_end_dates=analysis_chunk_end_dates,
            analysis_metric_1_alerts=analysis_metric_1_alerts,
            analysis_metric_2_alerts=analysis_metric_2_alerts,
            hover=hover,
            xaxis=x_axis,
            yaxis=y_axis,
            yaxis2=y_axis_2,
        )

    return figure


def _get_subplot_axes_names(index: int, y_axis_per_subplot: int = 2) -> Tuple:
    return tuple([f'x{index + 1}'] + [f'y{2 * index + 1 + a}' for a in range(y_axis_per_subplot)])


def _plot_compare_step_to_step(
    figure: Figure,
    metric_1_display_name: str,
    metric_2_display_name: str,
    analysis_metric_1: Union[np.ndarray, pd.Series],
    analysis_metric_2: Union[np.ndarray, pd.Series],
    reference_chunk_keys: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_periods: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_indices: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_start_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_end_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_metric_1: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_metric_2: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_chunk_keys: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_chunk_periods: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_chunk_indices: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_chunk_start_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_chunk_end_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_metric_1_alerts: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_metric_2_alerts: Optional[Union[np.ndarray, pd.Series]] = None,
    hover: Optional[Hover] = None,
    xaxis: Optional[str] = 'x',
    yaxis: Optional[str] = 'y',
    yaxis2: Optional[str] = 'y2',
) -> Figure:

    if figure is None:
        figure = Figure(
            title=f'{metric_1_display_name} versus {metric_2_display_name}',
            x_axis_title='Time'
            if is_time_based_x_axis(reference_chunk_start_dates, reference_chunk_end_dates)
            else 'Chunk',
            y_axis_title=f'{metric_1_display_name}',
            legend=dict(traceorder="grouped", itemclick=False, itemdoubleclick=False),
            height=500,
            yaxis2=dict(
                title=f"{metric_2_display_name}",
                anchor="x",
                overlaying="y",
                side="right",
            ),
        )

    has_reference_results = reference_chunk_indices is not None and len(reference_chunk_indices) > 0
    if has_reference_results and not is_time_based_x_axis(reference_chunk_start_dates, reference_chunk_end_dates):
        analysis_chunk_indices = analysis_chunk_indices + max(reference_chunk_indices) + 1  # type: ignore[arg-type]

    show_in_legend = xaxis == 'x1' and yaxis == 'y1'

    if has_reference_results:
        # region reference performance metric

        _hover = hover or Hover(
            template='%{period}<br />'
            'Chunk: <b>%{chunk_key}</b> &nbsp; &nbsp; %{x_coordinate} <br />'
            '%{metric_name}: <b>%{metric_value}</b><br />',
            # 'Sampling error range: +/- <b>%{sampling_error}</b><br />'
            show_extra=True,
        )

        _hover.add(np.asarray([metric_1_display_name] * len(analysis_metric_1)), 'metric_name')

        if reference_chunk_periods is not None:
            _hover.add(render_period_string(reference_chunk_periods), name='period')

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
        _hover.add(np.round(reference_metric_1, 4), name='metric_value')  # type: ignore[arg-type]

        figure.add_metric(
            data=reference_metric_1,
            indices=reference_chunk_indices,
            start_dates=reference_chunk_start_dates,
            end_dates=reference_chunk_end_dates,
            name='Estimated performance (reference)',
            color=Colors.INDIGO_PERSIAN.transparent(alpha=0.5),
            hover=_hover,
            line_dash='dash',
            xaxis=xaxis,
            yaxis=yaxis,
            showlegend=show_in_legend,
        )

        # endregion

        # region reference drift metric

        _hover = hover or Hover(
            template='%{period}<br />'
            'Chunk: <b>%{chunk_key}</b> &nbsp; &nbsp; %{x_coordinate} <br />'
            '%{metric_name}: <b>%{metric_value}</b><br />',
            # 'Sampling error range: +/- <b>%{sampling_error}</b><br />'
            show_extra=True,
        )
        _hover.add(
            np.asarray(["Reconstruction error"] * len(reference_metric_1)), 'metric_name'  # type: ignore[arg-type]
        )

        if reference_chunk_periods is not None:
            _hover.add(render_period_string(reference_chunk_periods), name='period')

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
        _hover.add(np.round(reference_metric_2, 4), name='metric_value')  # type: ignore[arg-type]

        figure.add_metric(
            data=reference_metric_2,
            indices=reference_chunk_indices,
            start_dates=reference_chunk_start_dates,
            end_dates=reference_chunk_end_dates,
            name='Reconstruction error (reference)',
            color=Colors.BLUE_SKY_CRAYOLA.transparent(alpha=0.5),
            xaxis=xaxis,
            yaxis=yaxis2,
            hover=_hover,
            showlegend=show_in_legend,
            # line_dash='dash',
        )

        figure.add_period_separator(
            x=(
                ensure_numpy(reference_chunk_indices)[0][-1] + 1
                if not is_time_based_x_axis(reference_chunk_start_dates, reference_chunk_end_dates)
                else ensure_numpy(analysis_chunk_start_dates)[0][0]  # type: ignore
            )
        )
        # endregion

    # region analysis performance metric

    _hover = hover or Hover(
        template='%{period} &nbsp; &nbsp; %{alert} <br />'
        'Chunk: <b>%{chunk_key}</b> &nbsp; &nbsp; %{x_coordinate} <br />'
        '%{metric_name}: <b>%{metric_value}</b><br />',
        # 'Sampling error range: +/- <b>%{sampling_error}</b><br />'
        show_extra=True,
    )
    _hover.add(np.asarray([metric_1_display_name] * len(analysis_metric_1)), 'metric_name')

    if analysis_chunk_periods is not None:
        _hover.add(render_period_string(analysis_chunk_periods), name='period')

    if analysis_metric_1_alerts is not None:
        _hover.add(render_alert_string(analysis_metric_1_alerts), name='alert')

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
    _hover.add(np.round(analysis_metric_1, 4), name='metric_value')

    figure.add_metric(
        data=analysis_metric_1,
        indices=analysis_chunk_indices,
        start_dates=analysis_chunk_start_dates,
        end_dates=analysis_chunk_end_dates,
        name='Estimated performance (analysis)',
        color=Colors.INDIGO_PERSIAN,
        hover=_hover,
        line_dash='dash',
        xaxis=xaxis,
        yaxis=yaxis,
        showlegend=show_in_legend,
    )

    # endregion

    # region analysis multivariate drift metric
    _hover = hover or Hover(
        template='%{period} &nbsp; &nbsp; %{alert} <br />'
        'Chunk: <b>%{chunk_key}</b> &nbsp; &nbsp; %{x_coordinate} <br />'
        '%{metric_name}: <b>%{metric_value}</b><br />',
        # 'Sampling error range: +/- <b>%{sampling_error}</b><br />'
        show_extra=True,
    )
    _hover.add(np.asarray(["Reconstruction error"] * len(analysis_metric_1)), 'metric_name')

    if analysis_chunk_periods is not None:
        _hover.add(render_period_string(analysis_chunk_periods), name='period')

    if analysis_metric_1_alerts is not None:
        _hover.add(render_alert_string(analysis_metric_1_alerts), name='alert')

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
    _hover.add(np.round(analysis_metric_2, 4), name='metric_value')

    figure.add_metric(
        data=analysis_metric_2,
        indices=analysis_chunk_indices,
        start_dates=analysis_chunk_start_dates,
        end_dates=analysis_chunk_end_dates,
        name='Reconstruction error (analysis)',
        color=Colors.BLUE_SKY_CRAYOLA,
        xaxis=xaxis,
        yaxis=yaxis2,
        hover=_hover,
        showlegend=show_in_legend,
    )

    # endregion

    # region alerts

    if analysis_metric_1_alerts is not None:
        figure.add_alert(
            data=analysis_metric_1,
            alerts=analysis_metric_1_alerts,
            indices=analysis_chunk_indices,
            start_dates=analysis_chunk_start_dates,
            end_dates=analysis_chunk_end_dates,
            name='Alert',
            legendgroup='alert',
            plot_areas=False,
            showlegend=False,
            xaxis=xaxis,
            yaxis=yaxis,
        )

    if analysis_metric_2_alerts is not None:
        figure.add_alert(
            data=analysis_metric_2,
            alerts=analysis_metric_2_alerts,
            indices=analysis_chunk_indices,
            start_dates=analysis_chunk_start_dates,
            end_dates=analysis_chunk_end_dates,
            name='Alert',
            legendgroup='alert',
            plot_areas=False,
            showlegend=show_in_legend,
            xaxis=xaxis,
            yaxis=yaxis2,
        )

    # endregion

    return figure
