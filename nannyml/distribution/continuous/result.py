import math
from typing import List, Optional, Dict, Any, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from nannyml import Chunker
from nannyml._typing import Key
from nannyml.base import PerColumnResult
from nannyml.exceptions import InvalidArgumentsException
from nannyml.plots import is_time_based_x_axis, Hover, Colors, Figure
from nannyml.plots.components.joy_plot import joy, alert as joy_alert


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

    def plot(
        self, alerts: Optional[Union[np.ndarray, pd.Series]] = None, alert_name: Optional[str] = None, *args, **kwargs
    ) -> go.Figure:
        """
        Creates a "joyplot over time" visualization to illustrate continuous distribution changes over time.

        Parameters
        ----------
        alerts: Optional[Union[np.ndarray, pd.Series]], default=None
            An array or series containing boolean values that indicate if an alert occurred for a particular chunk.
            This will be used to change the visualization of that chunk.
            The alerts could be retrieved from a UnivariateDriftCalculator result for example.
        alert_name: Optional[str], default=None
            A name for the origin of the alerts to be displayed in every subplot title. Required when the "alerts"
            parameter was passed.
        """

        if alerts is not None and not alert_name:
            raise InvalidArgumentsException(
                'the "alert_name" parameter must be given when the "alerts" parameter is set.'
            )
        subplot_title_format = (
            '<b>{display_names[0]}</b> distribution (alerts for <b>' + alert_name + '</b>)'
            if alerts is not None
            else '<b>{display_names[0]}</b> distribution'
        )
        return plot_continuous_distribution(self, alerts, subplot_title_format=subplot_title_format)


def plot_continuous_distribution(
    result: Result,
    alerts: Optional[Union[np.ndarray, pd.Series]] = None,
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

    reference_alerts = alerts[: len(reference_result)].reset_index(drop=True) if alerts is not None else None
    analysis_alerts = alerts[-len(analysis_result) :].reset_index(drop=True) if alerts is not None else None

    for idx, key in enumerate(result.keys()):
        row = (idx // number_of_columns) + 1
        col = (idx % number_of_columns) + 1

        (column_name,) = key.properties

        analysis_chunk_start_dates = analysis_result.chunk_start_dates
        analysis_chunk_end_dates = analysis_result.chunk_end_dates
        x_axis_is_time_based = is_time_based_x_axis(analysis_chunk_start_dates, analysis_chunk_end_dates)

        figure = _plot_joyplot(
            figure=figure,
            row=row,
            col=col,
            metric_display_name='',
            reference_distributions=reference_result.to_df().loc[:, (column_name,)],
            reference_alerts=reference_alerts,
            reference_chunk_keys=reference_result.chunk_keys,
            reference_chunk_periods=reference_result.chunk_periods,
            reference_chunk_indices=reference_result.chunk_indices,
            reference_chunk_start_dates=reference_result.chunk_start_dates,
            reference_chunk_end_dates=reference_result.chunk_end_dates,
            analysis_distributions=analysis_result.to_df().loc[:, (column_name,)],
            analysis_alerts=analysis_alerts,
            analysis_chunk_keys=analysis_result.chunk_keys,
            analysis_chunk_periods=analysis_result.chunk_periods,
            analysis_chunk_indices=analysis_result.chunk_indices,
            analysis_chunk_start_dates=analysis_chunk_start_dates,
            analysis_chunk_end_dates=analysis_chunk_end_dates,
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
