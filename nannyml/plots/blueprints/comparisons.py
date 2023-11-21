#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import itertools
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from nannyml._typing import Result
from nannyml.exceptions import InvalidArgumentsException
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
    **kwargs,
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

        # TODO: move this logic to the `Result` and `Metric` level.
        #       This is just a quick and very dirty way to check the same "metric" is being plotted, e.g.
        #       estimated f1 and realized f1.
        #       We're now making the assumption that they will both use the same column name in results.
        is_same_metric = key_1.properties[0] == key_2.properties[0]
        x_axis, y_axis, y_axis_2 = _get_subplot_axes_names(idx, y_axis_per_subplot=2, use_single_y_axis=is_same_metric)
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
            **kwargs,
        )

    return figure


def _get_subplot_axes_names(index: int, y_axis_per_subplot: int = 2, use_single_y_axis: bool = False) -> Tuple:
    # TODO: when removing subplots later, the decision on using a single y-axis or not should be given in the place
    #       best suited to determine it: the top level `compare()` function that has access to `Result` and `Metric`
    #       instances (due to the `information expert` principle
    #       https://en.wikipedia.org/wiki/GRASP_(object-oriented_design)#Information_expert)
    #       The decision can be made there and then either passes along as a parameter to a single function or
    #       determines one of multiple plotting functions to be used.
    #
    """Returns the names of the single x and y axes given the index of the subplot we're in.

    When `use_single_y_axis` is set to true, all returned y axes will have the same name, causing all metrics
    to use a single shared y-axis.
    """

    return tuple(
        [f'x{index + 1}'] + [f'y{2 * index + 1 + (0 if use_single_y_axis else a)}' for a in range(y_axis_per_subplot)]
    )


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
    metric_1_color=Colors.BLUE_SKY_CRAYOLA,
    metric_2_color=Colors.BLUE_SKY_CRAYOLA,
    **kwargs,
) -> Figure:
    _metric_1_kwargs = {k.replace('metric_1_', ''): v for k, v in kwargs.items() if k.startswith('metric_1_')}
    _metric_2_kwargs = {k.replace('metric_2_', ''): v for k, v in kwargs.items() if k.startswith('metric_2_')}

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

    has_analysis_results = (
        analysis_metric_1 is not None
        and len(analysis_metric_1) > 0
        and analysis_metric_2 is not None
        and len(analysis_metric_2) > 0
    )

    if has_reference_results and not is_time_based_x_axis(reference_chunk_start_dates, reference_chunk_end_dates):
        assert reference_chunk_indices is not None
        analysis_chunk_indices = analysis_chunk_indices + max(reference_chunk_indices) + 1

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
            _hover.add(render_period_string(reference_chunk_periods, color=metric_1_color), name='period')

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
        _hover.add(np.round(reference_metric_1, 4), name='metric_value')

        figure.add_metric(
            data=reference_metric_1,
            indices=reference_chunk_indices,
            start_dates=reference_chunk_start_dates,
            end_dates=reference_chunk_end_dates,
            name=f'{_metric_1_display_name}',
            hover=_hover,
            xaxis=xaxis,
            yaxis=yaxis,
            showlegend=True,
            color=metric_1_color,
            **_metric_1_kwargs,
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
                showlegend=True,
                with_additional_endpoint=True,
                color=metric_1_color,
                **_metric_1_kwargs,
            )

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
            _hover.add(render_period_string(reference_chunk_periods, color=metric_2_color), name='period')

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
        _hover.add(np.round(reference_metric_2, 4), name='metric_value')

        figure.add_metric(
            data=reference_metric_2,
            indices=reference_chunk_indices,
            start_dates=reference_chunk_start_dates,
            end_dates=reference_chunk_end_dates,
            name=f'{_metric_2_display_name}',
            xaxis=xaxis,
            yaxis=yaxis2,
            hover=_hover,
            showlegend=True,
            color=metric_2_color,
            **_metric_2_kwargs,
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
                showlegend=True,
                with_additional_endpoint=True,
                color=metric_2_color,
                **_metric_2_kwargs,
            )

        # region Period separator
        if has_analysis_results:
            is_time_based = is_time_based_x_axis(reference_chunk_start_dates, reference_chunk_end_dates)

            figure.add_period_separator(
                x=(
                    ensure_numpy(reference_chunk_indices)[0][-1] + 1
                    if not is_time_based
                    else ensure_numpy(analysis_chunk_start_dates)[0][0]
                )
            )

            reference_period_text_x = (
                reference_chunk_indices.mean()  # type: ignore
                if not is_time_based
                else reference_chunk_start_dates.mean()  # type: ignore
            )

            figure.add_annotation(
                x=reference_period_text_x,
                xshift=10,
                yref='paper',
                y=1,
                text="Reference",
                showarrow=False,
            )

            analyis_period_text_x = (
                analysis_chunk_indices.mean()  # type: ignore
                if not is_time_based
                else analysis_chunk_start_dates.mean()  # type: ignore
            )

            figure.add_annotation(
                x=analyis_period_text_x,
                xshift=15,
                yref='paper',
                y=1,
                text="Analysis",
                showarrow=False,
            )
        # endregion

        # endregion

    if has_analysis_results:
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
            _hover.add(render_period_string(analysis_chunk_periods, color=metric_1_color), name='period')

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
            name=f'{_metric_1_display_name}',
            hover=_hover,
            xaxis=xaxis,
            yaxis=yaxis,
            showlegend=not has_reference_results,
            color=metric_1_color,
            **_metric_1_kwargs,
        )

        if (
            analysis_metric_1_upper_confidence_bounds is not None
            and analysis_metric_1_lower_confidence_bounds is not None
        ):
            figure.add_confidence_band(
                upper_confidence_boundaries=analysis_metric_1_upper_confidence_bounds,
                lower_confidence_boundaries=analysis_metric_1_lower_confidence_bounds,
                indices=analysis_chunk_indices,
                start_dates=analysis_chunk_start_dates,
                end_dates=analysis_chunk_end_dates,
                name='Confidence band',
                xaxis=xaxis,
                yaxis=yaxis,
                showlegend=not has_reference_results,
                with_additional_endpoint=True,
                color=metric_1_color,
            )

        # endregion

        # region analysis metric 2
        _hover = hover or Hover(
            template='%{period} &nbsp; &nbsp; %{alert} <br />'
            'Chunk: <b>%{chunk_key}</b> &nbsp; &nbsp; %{x_coordinate} <br />'
            '%{metric_name}: <b>%{metric_value}</b><br />',
            # 'Sampling error range: +/- <b>%{sampling_error}</b><br />'
            show_extra=True,
        )
        _hover.add(
            np.asarray([render_metric_display_name(metric_2_display_name)] * len(analysis_metric_2)), 'metric_name'
        )

        if analysis_chunk_periods is not None:
            _hover.add(render_period_string(analysis_chunk_periods, color=metric_2_color), name='period')

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
            name=f'{_metric_2_display_name}',
            xaxis=xaxis,
            yaxis=yaxis2,
            hover=_hover,
            showlegend=not has_reference_results,
            color=metric_2_color,
            **_metric_2_kwargs,
        )

        if (
            analysis_metric_2_upper_confidence_bounds is not None
            and analysis_metric_2_lower_confidence_bounds is not None
        ):
            figure.add_confidence_band(
                upper_confidence_boundaries=analysis_metric_2_upper_confidence_bounds,
                lower_confidence_boundaries=analysis_metric_2_lower_confidence_bounds,
                indices=analysis_chunk_indices,
                start_dates=analysis_chunk_start_dates,
                end_dates=analysis_chunk_end_dates,
                name='Confidence band',
                xaxis=xaxis,
                yaxis=yaxis2,
                showlegend=not has_reference_results,
                with_additional_endpoint=True,
                color=metric_2_color,
                **_metric_2_kwargs,
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
                showlegend=True,
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
                showlegend=False,
                xaxis=xaxis,
                yaxis=yaxis2,
            )

        # endregion

    return figure


def render_display_name(metric_display_name: Union[str, Tuple]):
    if not isinstance(metric_display_name, str):
        if len(metric_display_name) == 1:
            return f'<b>{metric_display_name[0]}</b>'
        elif len(metric_display_name) >= 2:
            return f'<b>{metric_display_name[1]}</b> ({metric_display_name[0]})'
        else:
            return ', '.join(metric_display_name)
    else:
        return metric_display_name


def render_metric_display_name(metric_display_name: Union[str, Tuple]):
    if not isinstance(metric_display_name, str):
        if len(metric_display_name) == 1:
            return f'<b>{metric_display_name[0]}</b>'
        elif len(metric_display_name) >= 2:
            return f'<b>{metric_display_name[1]}</b>'
    else:
        return f'<b>{metric_display_name}</b>'


class ResultCompareMixin:
    def compare(self, other: Result):
        return ResultComparison(
            self, other, title=self.get_title(other), plot_kwargs=_get_plot_kwargs(self, other)  # type: ignore
        )

    @property
    def titles(self) -> Dict[type, str]:
        from nannyml.data_quality.missing.result import Result as MissingValueResult
        from nannyml.data_quality.unseen.result import Result as UnseenValuesResult
        from nannyml.drift.multivariate.data_reconstruction import Result as DataReconstructionDriftResult
        from nannyml.drift.univariate import Result as UnivariateDriftResult
        from nannyml.performance_calculation import Result as RealizedPerformanceResult
        from nannyml.performance_estimation.confidence_based import Result as CBPEResult
        from nannyml.performance_estimation.direct_loss_estimation import Result as DLEResult
        from nannyml.stats.avg.result import Result as StatsAvgResult
        from nannyml.stats.count import Result as StatsCountResult
        from nannyml.stats.std import Result as StatsStdResult
        from nannyml.stats.sum import Result as StatsSumResult

        _titles: Dict[type, Any] = {
            UnivariateDriftResult: "Univariate drift",
            DataReconstructionDriftResult: "Multivariate drift",
            RealizedPerformanceResult: "Realized performance",
            CBPEResult: "Estimated performance (CBPE)",
            DLEResult: "Estimated performance (DLE)",
            MissingValueResult: "Missing Values",
            UnseenValuesResult: "Unseen Values",
            StatsAvgResult: "Statistics, Average",
            StatsCountResult: "Statistics, Count",
            StatsStdResult: "Statistics, Standard Deviation",
            StatsSumResult: "Statistics, Sum",
        }

        return _titles

    def get_title(self, other: Result):
        return f"<b>{self.titles[type(self)]}</b> vs. <b>{self.titles[type(other)]}</b>"


def _get_plot_kwargs(result: Result, other: Result) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if _is_estimated_result(result):
        kwargs['metric_1_color'] = Colors.BLUE_SKY_CRAYOLA if _is_estimated_result(other) else Colors.INDIGO_PERSIAN
        kwargs['metric_1_line_dash'] = 'dash'

    if _is_estimated_result(other):
        kwargs['metric_2_color'] = Colors.INDIGO_PERSIAN
        kwargs['metric_2_line_dash'] = 'dash'

    if not _is_estimated_result(result) and not _is_estimated_result(other):
        kwargs['metric_2_color'] = Colors.INDIGO_PERSIAN

    return kwargs


def _is_estimated_result(result: Result) -> bool:
    from nannyml.performance_estimation.confidence_based import Result as CBPEResult
    from nannyml.performance_estimation.direct_loss_estimation import Result as DLEResult

    return isinstance(result, (CBPEResult, DLEResult))


class ResultComparison:
    def __init__(self, result: Result, other: Result, plot_kwargs: Dict[str, Any], title: Optional[str] = None):
        if len(result.keys()) != 1 or len(result.keys()) != 1:
            raise InvalidArgumentsException(
                f"you're comparing {len(result.keys())} metrics to {len(result.keys())} "
                "metrics, but should only compare 1 to 1 at a time. Please filter your"
                "results first using `result.filter()`"
            )

        self.result = result
        self.other = other
        self.title = title
        self.plot_kwargs = plot_kwargs

    def plot(self) -> Figure:
        return plot_2d_compare_step_to_step(
            result_1=self.result, result_2=self.other, plot_title=self.title, **self.plot_kwargs
        )
