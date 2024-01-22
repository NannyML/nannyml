import math
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from nannyml import Chunker
from nannyml._typing import Key
from nannyml.base import PerColumnResult
from nannyml.drift.univariate.result import Result as DriftResult
from nannyml.exceptions import InvalidArgumentsException
from nannyml.plots import Colors, Figure, Hover, is_time_based_x_axis
from nannyml.plots.components.joy_plot import alert as joy_alert
from nannyml.plots.components.joy_plot import joy


class Result(PerColumnResult):
    def __init__(
        self,
        results_data: pd.DataFrame,
        column_names: List[str],
        timestamp_column_name: Optional[str],
        chunker: Chunker,
    ):
        super().__init__(results_data, column_names)

        self.timestamp_column_name = timestamp_column_name
        self.chunker = chunker

    def keys(self) -> List[Key]:
        return [Key(properties=(c,), display_names=(c,)) for c in self.column_names]

    def plot(self, drift_result: Optional[DriftResult] = None, *args, **kwargs) -> go.Figure:
        """
        Creates a "joyplot over time" visualization to illustrate continuous distribution changes over time.

        Parameters
        ----------
        drift_result: Optional[nannyml.drift.univariate.Result]
            The result of a univariate drift calculation. When set it will be used to lookup alerts that occurred for
            each column / drift method combination in the drift calculation result.
            For each of these combinations a distribution plot of the column will be rendered showing the alerts
            for each drift method.
            When the `drift_result` parameter is not set no alerts will be rendered on the distribution plots.
        """

        if drift_result and not isinstance(drift_result, DriftResult):
            raise InvalidArgumentsException(
                'currently the alerts_from parameter only supports results of the ' 'UnivariateDriftCalculator.'
            )

        if drift_result:
            self.check_is_compatible_with(drift_result)

        return (
            _plot_continuous_distribution_with_alerts(self, drift_result)
            if drift_result
            else _plot_continuous_distribution(self)
        )

    def check_is_compatible_with(self, drift_result: DriftResult):
        # Check if all distribution columns are present in the drift result
        drift_column_names = set([col for tup in drift_result.keys() for col, _ in tup])
        distribution_column_names = set(self.column_names)

        missing_columns = distribution_column_names.difference(drift_column_names)
        if len(missing_columns) > 0:
            raise InvalidArgumentsException(
                "cannot render distribution plots with warnings. Following columns are not "
                f"in the drift results: {list(missing_columns)}"
            )

        # Check if both results use the same X-axis
        drift_result_is_time_based = is_time_based_x_axis(drift_result.chunk_start_dates, drift_result.chunk_end_dates)
        distr_result_is_time_based = is_time_based_x_axis(self.chunk_start_dates, self.chunk_end_dates)

        if drift_result_is_time_based != distr_result_is_time_based:
            raise InvalidArgumentsException(
                "cannot render distribution plots with warnings. Drift results are"
                f"{'' if drift_result_is_time_based else ' not'} time-based, distribution results are"
                f"{'' if distr_result_is_time_based else ' not'} time-based. Drift and distribution results should "
                f"both be time-based (have a timestamp column) or not."
            )


def _plot_continuous_distribution(
    result: Result,
    title: Optional[str] = 'Column distributions',
    figure: Optional[go.Figure] = None,
    x_axis_time_title: str = 'Time',
    x_axis_chunk_title: str = 'Chunk',
    y_axis_title: str = 'Values',
    figure_args: Optional[Dict[str, Any]] = None,
    subplot_title_format: str = '<b>{display_names[0]}</b> distribution',
    number_of_columns: Optional[int] = None,
) -> go.Figure:
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

        (column_name,) = key.properties

        figure = _plot_joyplot(
            figure=figure,
            row=row,
            col=col,
            metric_display_name='',
            reference_distributions=reference_result.to_df().loc[:, (column_name,)],
            reference_alerts=None,
            reference_chunk_keys=reference_result.chunk_keys,
            reference_chunk_periods=reference_result.chunk_periods,
            reference_chunk_indices=reference_result.chunk_indices,
            reference_chunk_start_dates=reference_result.chunk_start_dates,
            reference_chunk_end_dates=reference_result.chunk_end_dates,
            analysis_distributions=analysis_result.to_df().loc[:, (column_name,)],
            analysis_alerts=None,
            analysis_chunk_keys=analysis_result.chunk_keys,
            analysis_chunk_periods=analysis_result.chunk_periods,
            analysis_chunk_indices=analysis_result.chunk_indices,
            analysis_chunk_start_dates=analysis_result.chunk_start_dates,
            analysis_chunk_end_dates=analysis_result.chunk_end_dates,
        )

    return figure


def _plot_continuous_distribution_with_alerts(
    result: Result,
    drift_result: DriftResult,
    title: Optional[str] = 'Column distributions',
    figure: Optional[go.Figure] = None,
    x_axis_time_title: str = 'Time',
    x_axis_chunk_title: str = 'Chunk',
    y_axis_title: str = 'Values',
    figure_args: Optional[Dict[str, Any]] = None,
    subplot_title_format: str = '<b>{display_names[0]}</b> distribution (alerts for {display_names[1]})',
    number_of_columns: Optional[int] = None,
) -> go.Figure:
    number_of_plots = len(drift_result.keys())
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
                        subplot_title_format.format(display_names=key.display_names) for key in drift_result.keys()
                    ],
                ),
                **figure_args,
            )
        )

    reference_result = result.filter(period='reference')
    reference_result.data.sort_index(inplace=True)
    analysis_result = result.filter(period='analysis')
    analysis_result.data.sort_index(inplace=True)

    for idx, drift_key in enumerate(drift_result.keys()):
        row = (idx // number_of_columns) + 1
        col = (idx % number_of_columns) + 1

        (column_name, method_name) = drift_key.properties

        # reference_alerts = drift_result.filter(period='reference').alerts(drift_key)
        analysis_alerts = drift_result.filter(period='analysis').alerts(drift_key)

        figure = _plot_joyplot(
            figure=figure,
            row=row,
            col=col,
            metric_display_name='',
            reference_distributions=reference_result.to_df().xs(column_name, level=0, axis=1),
            reference_alerts=None,
            reference_chunk_keys=reference_result.chunk_keys,
            reference_chunk_periods=reference_result.chunk_periods,
            reference_chunk_indices=reference_result.chunk_indices,
            reference_chunk_start_dates=reference_result.chunk_start_dates,
            reference_chunk_end_dates=reference_result.chunk_end_dates,
            analysis_distributions=analysis_result.to_df().xs(column_name, level=0, axis=1),
            analysis_alerts=analysis_alerts,
            analysis_chunk_keys=analysis_result.chunk_keys,
            analysis_chunk_periods=analysis_result.chunk_periods,
            analysis_chunk_indices=analysis_result.chunk_indices,
            analysis_chunk_start_dates=analysis_result.chunk_start_dates,
            analysis_chunk_end_dates=analysis_result.chunk_end_dates,
        )

    return figure


def _plot_joyplot(
    figure: go.Figure,
    metric_display_name: str,
    reference_distributions: pd.DataFrame,
    analysis_distributions: pd.DataFrame,
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
) -> go.Figure:
    is_subplot = row is not None and col is not None
    subplot_args = dict(row=row, col=col) if is_subplot else None

    has_reference_results = reference_chunk_indices is not None and len(reference_chunk_indices) > 0

    if figure is None:
        figure = go.Figure(title='continuous distribution', x_axis_title='time', y_axis_title='value', height=500)

    if has_reference_results:  # TODO: move distribution calculations to calculator run
        figure = joy(
            fig=figure,
            data_distributions=reference_distributions,
            chunk_keys=reference_chunk_keys,
            chunk_indices=reference_chunk_indices,
            chunk_start_dates=reference_chunk_start_dates,
            chunk_end_dates=reference_chunk_end_dates,
            name='Reference',
            color=Colors.BLUE_SKY_CRAYOLA,
            subplot_args=subplot_args,
        )

        assert reference_chunk_indices is not None
        analysis_chunk_indices = analysis_chunk_indices + (max(reference_chunk_indices) + 1)

    figure = joy(
        fig=figure,
        data_distributions=analysis_distributions,
        chunk_keys=analysis_chunk_keys,
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
            chunk_keys=analysis_chunk_keys,
            chunk_indices=analysis_chunk_indices,
            chunk_start_dates=analysis_chunk_start_dates,
            chunk_end_dates=analysis_chunk_end_dates,
            subplot_args=subplot_args,
        )

    return figure
