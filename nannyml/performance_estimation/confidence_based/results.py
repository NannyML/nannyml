#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing CBPE estimation results and plotting implementations."""
import copy
import math
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from plotly import graph_objects as go

from nannyml._typing import ModelOutputsType, ProblemType
from nannyml.base import AbstractEstimatorResult
from nannyml.chunk import Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_estimation.confidence_based.metrics import Metric
from nannyml.plots import Colors, Figure, Hover, render_alert_string, render_period_string, render_x_coordinate

SUPPORTED_METRIC_VALUES = ['roc_auc', 'f1', 'precision', 'recall', 'specificity', 'accuracy']


class Result(AbstractEstimatorResult):
    """Contains results for CBPE estimation and adds plotting functionality."""

    def __init__(
        self,
        results_data: pd.DataFrame,
        metrics: List[Metric],
        y_pred: str,
        y_pred_proba: ModelOutputsType,
        y_true: str,
        chunker: Chunker,
        problem_type: ProblemType,
        timestamp_column_name: Optional[str] = None,
    ):
        super().__init__(results_data)

        self.metrics = metrics
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.y_true = y_true
        self.timestamp_column_name = timestamp_column_name
        self.problem_type = problem_type
        self.chunker = chunker

    def _filter(self, period: str, metrics: Optional[List[str]] = None, *args, **kwargs) -> AbstractEstimatorResult:
        if metrics is None:
            metrics = [metric.column_name for metric in self.metrics]

        data = pd.concat([self.data.loc[:, (['chunk'])], self.data.loc[:, (metrics,)]], axis=1)
        if period != 'all':
            data = data.loc[data.loc[:, ('chunk', 'period')] == period, :]

        data = data.reset_index(drop=True)
        res = copy.deepcopy(self)
        res.data = data
        res.metrics = [m for m in self.metrics if m.column_name in metrics]

        return res

    def plot(
        self,
        kind: str = 'performance',
        metric: Optional[Union[str, Metric]] = None,
        plot_reference: bool = False,
        *args,
        **kwargs,
    ) -> go.Figure:
        """Render plots based on CBPE estimation results.

        This function will return a :class:`plotly.graph_objects.Figure` object.
        The following kinds of plots are available:

        - ``performance``: a line plot rendering the estimated performance per :class:`~nannyml.chunk.Chunk` after
            applying the :meth:`~nannyml.performance_estimation.confidence_based.CBPE.calculate` method on a chunked
            dataset.

        Returns
        -------
        fig: :class:`plotly.graph_objs._figure.Figure`
            A :class:`~plotly.graph_objs._figure.Figure` object containing the requested drift plot.

            Can be saved to disk using the :meth:`~plotly.graph_objs._figure.Figure.write_image` method
            or shown rendered on screen using the :meth:`~plotly.graph_objs._figure.Figure.show` method.

        Examples
        --------
        >>> import nannyml as nml
        >>>
        >>> reference_df, analysis_df, target_df = nml.load_synthetic_binary_classification_dataset()
        >>>
        >>> estimator = nml.CBPE(
        >>>     y_true='work_home_actual',
        >>>     y_pred='y_pred',
        >>>     y_pred_proba='y_pred_proba',
        >>>     timestamp_column_name='timestamp',
        >>>     metrics=['f1', 'roc_auc']
        >>> )
        >>>
        >>> estimator.fit(reference_df)
        >>>
        >>> results = estimator.estimate(analysis_df)
        >>> results.plot().show()
        """
        if kind == 'performance':
            return self._plot_performance_metrics()
        else:
            raise InvalidArgumentsException(f"unknown plot kind '{kind}'. " f"Please provide on of: ['performance'].")

    def _plot_performance_metrics(self) -> Figure:
        number_of_columns = min(len(self.metrics), 2)

        fig = Figure(
            title='CBPE',
            x_axis_title='Time' if self.timestamp_column_name else 'Chunk',
            y_axis_title='Estimated metric',
            legend=dict(traceorder="grouped", itemclick=False, itemdoubleclick=False),
            height=len(self.metrics) * 500 / number_of_columns,
            subplot_args=dict(
                cols=number_of_columns,
                rows=math.ceil(len(self.metrics) / number_of_columns),
                subplot_titles=[f'Estimated <b>{metric.display_name}</b>' for metric in self.metrics],
            ),
        )

        for idx, metric in enumerate(self.metrics):
            row = (idx // number_of_columns) + 1
            col = (idx % number_of_columns) + 1

            show_in_legend = row == 1 and col == 1

            reference_results = self.filter(period='reference').to_df()
            has_reference_results = len(reference_results) > 0
            analysis_results = self.filter(period='analysis').to_df()

            # region reference metric

            if has_reference_results:
                hover = Hover(
                    template='%{period} &nbsp; &nbsp; %{alert}<br />'
                    'Chunk: <b>%{chunk_key}</b> &nbsp; &nbsp; %{x_coordinate}<br />'
                    f'{metric.display_name}: <b>%{{metric_value}}</b><br />'
                    'Sampling error range: +/- <b>%{sampling_error}</b><br />'
                    '<extra></extra>'
                )
                hover.add(render_period_string(reference_results[('chunk', 'period')]), name='period')
                hover.add(render_alert_string(reference_results[(metric.column_name, 'alert')]), name='alert')
                hover.add(reference_results[('chunk', 'key')], name='chunk_key')
                hover.add(
                    render_x_coordinate(
                        reference_results[('chunk', 'chunk_index')],
                        reference_results[('chunk', 'start_date')],
                        reference_results[('chunk', 'end_date')],
                    ),
                    name='x_coordinate',
                )
                hover.add(np.round(reference_results[(metric.column_name, 'value')], 4), name='metric_value')
                hover.add(
                    np.round(reference_results[(metric.column_name, 'sampling_error')] * 3, 4), name='sampling_error'
                )

                fig.add_metric(
                    data=reference_results[(metric.column_name, 'value')],
                    indices=reference_results[('chunk', 'chunk_index')],
                    start_dates=reference_results[('chunk', 'start_date')],
                    end_dates=reference_results[('chunk', 'end_date')],
                    name='Metric reference',
                    color=Colors.BLUE_SKY_CRAYOLA,
                    hover=hover,
                    subplot_args=dict(row=row, col=col),
                    legendgroup='metric_reference',
                    showlegend=show_in_legend,
                    # line_dash='dash',
                )
            # endregion

            # region analysis metrics

            hover = Hover(
                template='%{period} &nbsp; &nbsp; %{alert} <br />'
                'Chunk: <b>%{chunk_key}</b> &nbsp; &nbsp; %{x_coordinate} <br />'
                f'{metric.display_name}: <b>%{{metric_value}}</b><br />'
                'Sampling error range: +/- <b>%{sampling_error}</b><br />'
                '<extra></extra>'
            )
            hover.add(render_period_string(analysis_results[('chunk', 'period')]), name='period')
            hover.add(render_alert_string(analysis_results[(metric.column_name, 'alert')]), name='alert')
            hover.add(analysis_results[('chunk', 'key')], name='chunk_key')
            hover.add(
                render_x_coordinate(
                    analysis_results[('chunk', 'chunk_index')],
                    analysis_results[('chunk', 'start_date')],
                    analysis_results[('chunk', 'end_date')],
                ),
                name='x_coordinate',
            )
            hover.add(np.round(analysis_results[(metric.column_name, 'value')], 4), name='metric_value')
            hover.add(np.round(analysis_results[(metric.column_name, 'sampling_error')] * 3, 4), name='sampling_error')

            analysis_indices = analysis_results[('chunk', 'chunk_index')]
            if has_reference_results:
                analysis_indices += +max(reference_results[('chunk', 'chunk_index')]) + 1

            fig.add_metric(
                data=analysis_results[(metric.column_name, 'value')],
                indices=analysis_indices,
                start_dates=analysis_results[('chunk', 'start_date')],
                end_dates=analysis_results[('chunk', 'end_date')],
                name='Metric (analysis)',
                color=Colors.INDIGO_PERSIAN,
                hover=hover,
                subplot_args=dict(row=row, col=col),
                legendgroup='metric_analysis',
                showlegend=show_in_legend,
                # line_dash='dash',
            )
            # endregion

            # region alert

            fig.add_alert(
                data=analysis_results[(metric.column_name, 'value')],
                alerts=analysis_results[(metric.column_name, 'alert')],
                indices=analysis_indices,
                start_dates=analysis_results[('chunk', 'start_date')],
                end_dates=analysis_results[('chunk', 'end_date')],
                name='Alert',
                subplot_args=dict(row=row, col=col),
                legendgroup='alert',
                plot_areas=False,
                showlegend=show_in_legend,
            )

            # endregion

            # region thresholds

            if has_reference_results:
                fig.add_threshold(
                    data=reference_results[(metric.column_name, 'upper_threshold')],
                    indices=reference_results[('chunk', 'chunk_index')],
                    start_dates=reference_results[('chunk', 'start_date')],
                    end_dates=reference_results[('chunk', 'end_date')],
                    name='Threshold',
                    with_additional_endpoint=True,
                    subplot_args=dict(row=row, col=col),
                    legendgroup='thresh',
                    showlegend=False,
                )

                fig.add_threshold(
                    data=reference_results[(metric.column_name, 'lower_threshold')],
                    indices=reference_results[('chunk', 'chunk_index')],
                    start_dates=reference_results[('chunk', 'start_date')],
                    end_dates=reference_results[('chunk', 'end_date')],
                    name='Threshold',
                    with_additional_endpoint=True,
                    subplot_args=dict(row=row, col=col),
                    legendgroup='thresh',
                    showlegend=True,
                )

            fig.add_threshold(
                data=analysis_results[(metric.column_name, 'upper_threshold')],
                indices=analysis_indices,
                start_dates=analysis_results[('chunk', 'start_date')],
                end_dates=analysis_results[('chunk', 'end_date')],
                name='Threshold',
                with_additional_endpoint=True,
                subplot_args=dict(row=row, col=col),
                legendgroup='thresh',
                showlegend=False,
            )

            fig.add_threshold(
                data=analysis_results[(metric.column_name, 'lower_threshold')],
                indices=analysis_indices,
                start_dates=analysis_results[('chunk', 'start_date')],
                end_dates=analysis_results[('chunk', 'end_date')],
                name='Threshold',
                with_additional_endpoint=True,
                subplot_args=dict(row=row, col=col),
                legendgroup='thresh',
                showlegend=False,
            )
            # endregion

            # region confidence bands

            if has_reference_results:
                fig.add_confidence_band(
                    upper_confidence_boundaries=reference_results[(metric.column_name, 'upper_confidence_boundary')],
                    lower_confidence_boundaries=reference_results[(metric.column_name, 'lower_confidence_boundary')],
                    indices=reference_results[('chunk', 'chunk_index')],
                    start_dates=reference_results[('chunk', 'start_date')],
                    end_dates=reference_results[('chunk', 'end_date')],
                    name='Sampling error',
                    color=Colors.BLUE_SKY_CRAYOLA,
                    with_additional_endpoint=True,
                    subplot_args=dict(row=row, col=col),
                    showlegend=show_in_legend,
                )

            fig.add_confidence_band(
                upper_confidence_boundaries=analysis_results[(metric.column_name, 'upper_confidence_boundary')],
                lower_confidence_boundaries=analysis_results[(metric.column_name, 'lower_confidence_boundary')],
                indices=analysis_indices,
                start_dates=analysis_results[('chunk', 'start_date')],
                end_dates=analysis_results[('chunk', 'end_date')],
                name='Sampling error',
                color=Colors.INDIGO_PERSIAN,
                with_additional_endpoint=True,
                subplot_args=dict(row=row, col=col),
                showlegend=show_in_legend,
            )
            # endregion

            if has_reference_results:
                fig.add_period_separator(
                    x=(
                        self.filter(period='reference').to_df()[('chunk', 'chunk_index')].iloc[-1] + 1
                        if self.timestamp_column_name is None
                        else self.filter(period='analysis').to_df()[('chunk', 'start_date')].iloc[0]
                    ),
                    subplot_args=dict(row=row, col=col),
                )

        return fig
