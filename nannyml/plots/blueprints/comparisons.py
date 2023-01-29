#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import itertools
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from nannyml._typing import Result
from nannyml.plots import Colors
from nannyml.plots.components import Figure, Hover, render_alert_string, render_period_string, render_x_coordinate
from nannyml.plots.util import ensure_numpy, is_time_based_x_axis


def plot_2d_compare_step_to_step(
    result_1: Result,
    result_2: Result,
    plot_title: Optional[str] = None,
    x_axis_time_title: str = 'Time',
    x_axis_chunk_title: str = 'Chunk',
    y_axis_title: str = 'Comparison',
    subplot_titles: Optional[List[str]] = None,
    number_of_columns: Optional[int] = None,
    hover: Optional[Hover] = None,
) -> Figure:

    # validate if both result keysets are compatible for plotting
    items = list(itertools.product(result_1.keys(), result_2.keys()))

    number_of_plots = len(items)
    if number_of_columns is None:
        number_of_columns = min(number_of_plots, 1)
    number_of_rows = math.ceil(number_of_plots / number_of_columns)

    reference_result_1: Result = result_1.filter(period='reference')
    analysis_result_1: Result = result_1.filter(period='analysis')
    reference_result_2: Result = result_2.filter(period='reference')
    analysis_result_2: Result = result_2.filter(period='analysis')

    reference_chunk_indices = reference_result_1.chunk_indices
    reference_chunk_start_dates = reference_result_1.chunk_start_dates
    reference_chunk_end_dates = reference_result_1.chunk_end_dates
    reference_chunk_periods = reference_result_1.chunk_periods
    reference_chunk_keys = reference_result_1.chunk_keys

    analysis_chunk_indices = analysis_result_1.chunk_indices
    analysis_chunk_start_dates = analysis_result_1.chunk_start_dates
    analysis_chunk_end_dates = analysis_result_1.chunk_end_dates
    analysis_chunk_periods = analysis_result_1.chunk_periods
    analysis_chunk_keys = analysis_result_1.chunk_keys

    # region setup axes

    x_axis_title = (
        x_axis_time_title
        if is_time_based_x_axis(reference_chunk_start_dates, reference_chunk_end_dates)
        else x_axis_chunk_title
    )
    figure = Figure(
        plot_title or '',
        x_axis_title,
        y_axis_title,
        legend=dict(traceorder="grouped", itemclick=False, itemdoubleclick=False),
        height=number_of_plots * 500 / number_of_columns,
    )

    subplot_specs = [[{"secondary_y": True} for _ in range(number_of_columns)] for _ in range(number_of_rows)]
    if subplot_titles is None:
        subplot_titles = [
            f'{render_display_name(key_1.display_names)} vs. {render_display_name(key_2.display_names)}'
            for key_1, key_2 in items
        ]
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

    for idx, (key_1, key_2) in enumerate(items):
        reference_metric_1 = reference_result_1.values(key_1)
        reference_metric_2 = reference_result_2.values(key_2)
        reference_metric_1_upper_confidence_bounds = reference_result_1.upper_confidence_bounds(key_1)
        reference_metric_1_lower_confidence_bounds = reference_result_1.lower_confidence_bounds(key_1)
        reference_metric_2_upper_confidence_bounds = reference_result_2.upper_confidence_bounds(key_2)
        reference_metric_2_lower_confidence_bounds = reference_result_2.lower_confidence_bounds(key_2)
        analysis_metric_1 = analysis_result_1.values(key_1)
        analysis_metric_2 = analysis_result_2.values(key_2)
        analysis_metric_1_alerts = analysis_result_1.alerts(key_1)
        analysis_metric_2_alerts = analysis_result_2.alerts(key_2)
        analysis_metric_1_upper_confidence_bounds = analysis_result_1.upper_confidence_bounds(key_1)
        analysis_metric_1_lower_confidence_bounds = analysis_result_1.lower_confidence_bounds(key_1)
        analysis_metric_2_upper_confidence_bounds = analysis_result_2.upper_confidence_bounds(key_2)
        analysis_metric_2_lower_confidence_bounds = analysis_result_2.lower_confidence_bounds(key_2)

        x_axis, y_axis, y_axis_2 = _get_subplot_axes_names(idx, y_axis_per_subplot=2)
        _set_y_axis_title(figure, y_axis, render_metric_display_name(key_1.display_names))
        _set_y_axis_title(figure, y_axis_2, render_metric_display_name(key_2.display_names))

        figure = _plot_compare_step_to_step(
            figure=figure,
            metric_1_display_name=key_1.display_names,
            metric_2_display_name=key_2.display_names,
            analysis_metric_1=analysis_metric_1,
            analysis_metric_2=analysis_metric_2,
            reference_chunk_keys=reference_chunk_keys,
            reference_chunk_periods=reference_chunk_periods,
            reference_chunk_indices=reference_chunk_indices,
            reference_chunk_start_dates=reference_chunk_start_dates,
            reference_chunk_end_dates=reference_chunk_end_dates,
            reference_metric_1=reference_metric_1,
            reference_metric_2=reference_metric_2,
            reference_metric_1_upper_confidence_bounds=reference_metric_1_upper_confidence_bounds,
            reference_metric_1_lower_confidence_bounds=reference_metric_1_lower_confidence_bounds,
            reference_metric_2_upper_confidence_bounds=reference_metric_2_upper_confidence_bounds,
            reference_metric_2_lower_confidence_bounds=reference_metric_2_lower_confidence_bounds,
            analysis_chunk_keys=analysis_chunk_keys,
            analysis_chunk_periods=analysis_chunk_periods,
            analysis_chunk_indices=analysis_chunk_indices,
            analysis_chunk_start_dates=analysis_chunk_start_dates,
            analysis_chunk_end_dates=analysis_chunk_end_dates,
            analysis_metric_1_alerts=analysis_metric_1_alerts,
            analysis_metric_2_alerts=analysis_metric_2_alerts,
            analysis_metric_1_upper_confidence_bounds=analysis_metric_1_upper_confidence_bounds,
            analysis_metric_1_lower_confidence_bounds=analysis_metric_1_lower_confidence_bounds,
            analysis_metric_2_upper_confidence_bounds=analysis_metric_2_upper_confidence_bounds,
            analysis_metric_2_lower_confidence_bounds=analysis_metric_2_lower_confidence_bounds,
            hover=hover,
            xaxis=x_axis,
            yaxis=y_axis,
            yaxis2=y_axis_2,
        )

    return figure


def _get_subplot_axes_names(index: int, y_axis_per_subplot: int = 2) -> Tuple:
    return tuple([f'x{index + 1}'] + [f'y{2 * index + 1 + a}' for a in range(y_axis_per_subplot)])


def _set_y_axis_title(figure: Figure, y_axis_name: str, title: str):
    y_name = y_axis_name[0] + 'axis' + y_axis_name[1 : len(y_axis_name)]
    figure.layout.__getattr__(y_name).title = title


def _plot_compare_step_to_step(  # noqa: C901
    figure: Figure,
    metric_1_display_name: Union[str, Tuple],
    metric_2_display_name: Union[str, Tuple],
    analysis_metric_1: Union[np.ndarray, pd.Series],
    analysis_metric_2: Union[np.ndarray, pd.Series],
    reference_chunk_keys: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_periods: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_indices: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_start_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_chunk_end_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_metric_1: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_metric_2: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_metric_1_upper_confidence_bounds: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_metric_1_lower_confidence_bounds: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_metric_2_upper_confidence_bounds: Optional[Union[np.ndarray, pd.Series]] = None,
    reference_metric_2_lower_confidence_bounds: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_chunk_keys: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_chunk_periods: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_chunk_indices: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_chunk_start_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_chunk_end_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_metric_1_alerts: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_metric_2_alerts: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_metric_1_upper_confidence_bounds: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_metric_1_lower_confidence_bounds: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_metric_2_upper_confidence_bounds: Optional[Union[np.ndarray, pd.Series]] = None,
    analysis_metric_2_lower_confidence_bounds: Optional[Union[np.ndarray, pd.Series]] = None,
    hover: Optional[Hover] = None,
    xaxis: Optional[str] = 'x',
    yaxis: Optional[str] = 'y',
    yaxis2: Optional[str] = 'y2',
) -> Figure:
    _metric_1_display_name = render_display_name(metric_1_display_name)
    _metric_2_display_name = render_display_name(metric_2_display_name)

    if figure is None:
        figure = Figure(
            title=f'{_metric_1_display_name} versus {_metric_2_display_name}',
            x_axis_title='Time'
            if is_time_based_x_axis(reference_chunk_start_dates, reference_chunk_end_dates)
            else 'Chunk',
            y_axis_title=f'{_metric_1_display_name}',
            legend=dict(traceorder="grouped", itemclick=False, itemdoubleclick=False),
            height=500,
            yaxis2=dict(
                title=f"{_metric_2_display_name}",
                anchor="x",
                overlaying="y",
                side="right",
            ),
        )

    has_reference_results = (
        reference_metric_1 is not None
        and len(reference_metric_1) > 0
        and reference_metric_2 is not None
        and len(reference_metric_2) > 0
    )
    if has_reference_results and not is_time_based_x_axis(reference_chunk_start_dates, reference_chunk_end_dates):
        analysis_chunk_indices = analysis_chunk_indices + max(reference_chunk_indices) + 1  # type: ignore[arg-type]

    show_in_legend = xaxis == 'x1' and yaxis == 'y1'

    confidence_band_in_legend = True

    if has_reference_results:
        # region reference metric 1

        _hover = hover or Hover(
            template='%{period}<br />'
            'Chunk: <b>%{chunk_key}</b> &nbsp; &nbsp; %{x_coordinate} <br />'
            '%{metric_name}: <b>%{metric_value}</b><br />',
            # 'Sampling error range: +/- <b>%{sampling_error}</b><br />'
            show_extra=True,
        )

        assert reference_metric_1 is not None
        _hover.add(np.asarray([_metric_1_display_name] * len(reference_metric_1)), 'metric_name')

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
            name=f'{_metric_1_display_name} (reference)',
            color=Colors.INDIGO_PERSIAN.transparent(alpha=0.5),
            hover=_hover,
            line_dash='dash',
            xaxis=xaxis,
            yaxis=yaxis,
            showlegend=show_in_legend,
        )

        if (
            reference_metric_1_upper_confidence_bounds is not None
            and reference_metric_1_lower_confidence_bounds is not None
        ):
            figure.add_confidence_band(
                upper_confidence_boundaries=reference_metric_1_upper_confidence_bounds,
                lower_confidence_boundaries=reference_metric_1_lower_confidence_bounds,
                indices=reference_chunk_indices,
                start_dates=reference_chunk_start_dates,
                end_dates=reference_chunk_end_dates,
                name='Confidence band',
                xaxis=xaxis,
                yaxis=yaxis,
                showlegend=show_in_legend and confidence_band_in_legend,
                with_additional_endpoint=True,
            )
            confidence_band_in_legend = False

        # endregion

        # region reference metric 2

        _hover = hover or Hover(
            template='%{period}<br />'
            'Chunk: <b>%{chunk_key}</b> &nbsp; &nbsp; %{x_coordinate} <br />'
            '%{metric_name}: <b>%{metric_value}</b><br />',
            # 'Sampling error range: +/- <b>%{sampling_error}</b><br />˚'
            show_extra=True,
        )

        assert reference_metric_2 is not None
        _hover.add(
            np.asarray([render_metric_display_name(metric_2_display_name)] * len(reference_metric_2)), 'metric_name'
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
            name=f'{_metric_2_display_name} (reference)',
            color=Colors.BLUE_SKY_CRAYOLA.transparent(alpha=0.5),
            xaxis=xaxis,
            yaxis=yaxis2,
            hover=_hover,
            showlegend=show_in_legend,
            # line_dash='dash',
        )

        if (
            reference_metric_2_upper_confidence_bounds is not None
            and reference_metric_2_lower_confidence_bounds is not None
        ):
            figure.add_confidence_band(
                upper_confidence_boundaries=reference_metric_2_upper_confidence_bounds,
                lower_confidence_boundaries=reference_metric_2_lower_confidence_bounds,
                indices=reference_chunk_indices,
                start_dates=reference_chunk_start_dates,
                end_dates=reference_chunk_end_dates,
                name='Confidence band',
                xaxis=xaxis,
                yaxis=yaxis2,
                showlegend=show_in_legend and confidence_band_in_legend,
                with_additional_endpoint=True,
            )
            confidence_band_in_legend = False

        figure.add_period_separator(
            x=(
                ensure_numpy(reference_chunk_indices)[0][-1] + 1
                if not is_time_based_x_axis(reference_chunk_start_dates, reference_chunk_end_dates)
                else ensure_numpy(analysis_chunk_start_dates)[0][0]  # type: ignore
            )
        )
        # endregion

    # region analysis metric 1

    _hover = hover or Hover(
        template='%{period} &nbsp; &nbsp; %{alert} <br />'
        'Chunk: <b>%{chunk_key}</b> &nbsp; &nbsp; %{x_coordinate} <br />'
        '%{metric_name}: <b>%{metric_value}</b><br />',
        # 'Sampling error range: +/- <b>%{sampling_error}</b><br />'
        show_extra=True,
    )
    _hover.add(np.asarray([_metric_1_display_name] * len(analysis_metric_1)), 'metric_name')

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
        name=f'{_metric_1_display_name} (analysis)',
        color=Colors.INDIGO_PERSIAN,
        hover=_hover,
        line_dash='dash',
        xaxis=xaxis,
        yaxis=yaxis,
        showlegend=show_in_legend,
    )

    if analysis_metric_1_upper_confidence_bounds is not None and analysis_metric_1_lower_confidence_bounds is not None:
        figure.add_confidence_band(
            upper_confidence_boundaries=analysis_metric_1_upper_confidence_bounds,
            lower_confidence_boundaries=analysis_metric_1_lower_confidence_bounds,
            indices=analysis_chunk_indices,
            start_dates=analysis_chunk_start_dates,
            end_dates=analysis_chunk_end_dates,
            name='Confidence band',
            xaxis=xaxis,
            yaxis=yaxis,
            showlegend=show_in_legend and confidence_band_in_legend,
            with_additional_endpoint=True,
        )
        confidence_band_in_legend = False

    # endregion

    # region analysis metric 2
    _hover = hover or Hover(
        template='%{period} &nbsp; &nbsp; %{alert} <br />'
        'Chunk: <b>%{chunk_key}</b> &nbsp; &nbsp; %{x_coordinate} <br />'
        '%{metric_name}: <b>%{metric_value}</b><br />',
        # 'Sampling error range: +/- <b>%{sampling_error}</b><br />'
        show_extra=True,
    )
    _hover.add(np.asarray([render_metric_display_name(metric_2_display_name)] * len(analysis_metric_2)), 'metric_name')

    if analysis_chunk_periods is not None:
        _hover.add(render_period_string(analysis_chunk_periods), name='period')

    if analysis_metric_2_alerts is not None:
        _hover.add(render_alert_string(analysis_metric_2_alerts), name='alert')

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
        name=f'{_metric_2_display_name} (analysis)',
        color=Colors.BLUE_SKY_CRAYOLA,
        xaxis=xaxis,
        yaxis=yaxis2,
        hover=_hover,
        showlegend=show_in_legend,
    )

    if analysis_metric_2_upper_confidence_bounds is not None and analysis_metric_2_lower_confidence_bounds is not None:
        figure.add_confidence_band(
            upper_confidence_boundaries=analysis_metric_2_upper_confidence_bounds,
            lower_confidence_boundaries=analysis_metric_2_lower_confidence_bounds,
            indices=analysis_chunk_indices,
            start_dates=analysis_chunk_start_dates,
            end_dates=analysis_chunk_end_dates,
            name='Confidence band',
            xaxis=xaxis,
            yaxis=yaxis2,
            showlegend=show_in_legend and confidence_band_in_legend,
            with_additional_endpoint=True,
        )
        confidence_band_in_legend = False

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


def render_display_name(metric_display_name: Union[str, Tuple]):
    if not isinstance(metric_display_name, str):
        if len(metric_display_name) == 1:
            return f'<b>{metric_display_name[0]}</b>'
        elif len(metric_display_name) == 2:
            return f'<b>{metric_display_name[1]}</b> ({metric_display_name[0]})'
        else:
            return ', '.join(metric_display_name)
    else:
        return metric_display_name


def render_metric_display_name(metric_display_name: Union[str, Tuple]):
    if not isinstance(metric_display_name, str):
        if len(metric_display_name) == 1:
            return f'<b>{metric_display_name[0]}</b>'
        elif len(metric_display_name) == 2:
            return f'<b>{metric_display_name[1]}</b>'
    else:
        return f'<b>{metric_display_name}</b>'


class ResultCompareMixin:
    def compare(self, other: Result):
        return ResultComparison(self, other, title=self._get_title(other))  # type: ignore

    def _get_title(self, other: Result):
        from nannyml.drift.multivariate.data_reconstruction import Result as DataReconstructionDriftResult
        from nannyml.drift.univariate import Result as UnivariateDriftResult
        from nannyml.performance_calculation import Result as RealizedPerformanceResult
        from nannyml.performance_estimation.confidence_based import Result as CBPEResult
        from nannyml.performance_estimation.direct_loss_estimation import Result as DLEResult

        _result_title_names: Dict[type, Any] = {
            UnivariateDriftResult: "Univariate drift",
            DataReconstructionDriftResult: "Multivariate drift",
            RealizedPerformanceResult: "Realized performance",
            CBPEResult: "Estimated performance (CBPE)",
            DLEResult: "Estimated performance (DLE)",
        }

        return f"<b>{_result_title_names[type(self)]}</b> vs. <b>{_result_title_names[type(other)]}</b>"


class ResultComparison:
    def __init__(self, result: Result, other: Result, title: Optional[str] = None):
        self.result = result
        self.other = other
        self.title = title

    def plot(self) -> Figure:
        return plot_2d_compare_step_to_step(
            result_1=self.result,
            result_2=self.other,
            plot_title=self.title,
        )
