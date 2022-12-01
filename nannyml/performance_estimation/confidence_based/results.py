#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing CBPE estimation results and plotting implementations."""
import copy
from typing import List, Optional, Union

import pandas as pd
from plotly import graph_objects as go

from nannyml._typing import ModelOutputsType, ProblemType
from nannyml.base import AbstractEstimatorResult
from nannyml.chunk import Chunker
from nannyml.drift.multivariate.data_reconstruction import Result as MultivariateResult
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_estimation.confidence_based.metrics import Metric
from nannyml.plots import Colors, is_time_based_x_axis
from nannyml.plots.blueprints.metrics import plot_metric_list
from nannyml.plots.components import Figure
from nannyml.usage_logging import UsageEvent, log_usage

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

    @log_usage(UsageEvent.CBPE_PLOT, metadata_from_kwargs=['kind'])
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
            return plot_metric_list(
                self, title='Estimated performance <b>(CBPE)</b>', subplot_title_format='Estimated <b>{metric_name}</b>'
            )
        else:
            raise InvalidArgumentsException(f"unknown plot kind '{kind}'. " f"Please provide on of: ['performance'].")

    def compare(self, result: Union[MultivariateResult]):
        if isinstance(result, MultivariateResult):
            return ResultMultivariateComparison(performance_result=self, multivariate_result=result)


class ResultMultivariateComparison:
    def __init__(self, performance_result: Result, multivariate_result: MultivariateResult):
        self.performance_result = performance_result
        self.multivariate_result = multivariate_result
        if len(self.performance_result.metrics) != 1:
            raise InvalidArgumentsException(
                f"performance result contains {len(performance_result.metrics)} metrics. "
                "Please use the 'filter' method to keep just one."
            )
        self.performance_metric = self.performance_result.metrics[0]

    def plot(self) -> Figure:
        reference_performance_result = self.performance_result.filter(period='reference').to_df()
        analysis_performance_result = self.performance_result.filter(period='analysis').to_df()

        reference_multivariate_result = self.multivariate_result.filter(period='reference').to_df()
        analysis_multivariate_result = self.multivariate_result.filter(period='analysis').to_df()

        reference_chunk_indices = reference_performance_result[('chunk', 'chunk_index')]
        reference_chunk_start_dates = reference_performance_result[('chunk', 'start_date')]
        reference_chunk_end_dates = reference_performance_result[('chunk', 'end_date')]
        reference_performance_metric = reference_performance_result[(self.performance_metric.column_name, 'value')]
        reference_multivariate_metric = reference_multivariate_result[('reconstruction_error', 'value')]

        analysis_chunk_indices = analysis_performance_result[('chunk', 'chunk_index')]
        analysis_chunk_start_dates = analysis_performance_result[('chunk', 'start_date')]
        analysis_chunk_end_dates = analysis_performance_result[('chunk', 'end_date')]
        analysis_performance_metric = analysis_performance_result[(self.performance_metric.column_name, 'value')]
        analysis_multivariate_metric = analysis_multivariate_result[('reconstruction_error', 'value')]

        analysis_performance_alerts = analysis_performance_result[(self.performance_metric.column_name, 'alert')]
        analysis_multivariate_alerts = analysis_multivariate_result[('reconstruction_error', 'alert')]

        figure = Figure(
            title='CBPE',
            x_axis_title='Time'
            if is_time_based_x_axis(reference_chunk_start_dates, reference_chunk_end_dates)
            else 'Chunk',
            y_axis_title='Estimated metric',
            legend=dict(traceorder="grouped", itemclick=False, itemdoubleclick=False),
            height=500,
        )

        has_reference_results = reference_chunk_indices is not None and len(reference_chunk_indices) > 0

        if has_reference_results:
            figure.add_metric(
                data=reference_performance_metric,
                indices=reference_chunk_indices,
                start_dates=reference_chunk_start_dates,
                end_dates=reference_chunk_end_dates,
                name=f'Estimated {self.performance_metric.display_name} (reference)',
                color=Colors.INDIGO_PERSIAN,
                # hover=_hover,
                line_dash='dash',
            )

            figure.add_metric(
                data=reference_multivariate_metric,
                indices=reference_chunk_indices,
                start_dates=reference_chunk_start_dates,
                end_dates=reference_chunk_end_dates,
                name='Reconstruction error (reference)',
                color=Colors.BLUE_SKY_CRAYOLA,
                # hover=_hover,
                # line_dash='dash',
            )

            figure.add_period_separator(
                x=(
                    reference_chunk_indices[-1] + 1
                    if not is_time_based_x_axis(reference_chunk_start_dates, reference_chunk_end_dates)
                    else analysis_chunk_start_dates[0]  # type: ignore
                )
            )

        figure.add_metric(
            data=analysis_performance_metric,
            indices=analysis_chunk_indices,
            start_dates=analysis_chunk_start_dates,
            end_dates=analysis_chunk_end_dates,
            name=f'Estimated {self.performance_metric.display_name} (analysis)',
            color=Colors.INDIGO_PERSIAN,
            # hover=_hover,
            line_dash='dash',
        )

        figure.add_metric(
            data=analysis_multivariate_metric,
            indices=analysis_chunk_indices,
            start_dates=analysis_chunk_start_dates,
            end_dates=analysis_chunk_end_dates,
            name='Reconstruction error (analysis)',
            color=Colors.BLUE_SKY_CRAYOLA,
            # hover=_hover,
            # line_dash='dash',
        )

        if analysis_performance_alerts is not None:
            figure.add_alert(
                data=analysis_multivariate_metric,
                alerts=analysis_multivariate_alerts,
                indices=analysis_chunk_indices,
                start_dates=analysis_chunk_start_dates,
                end_dates=analysis_chunk_end_dates,
                name='Alert',
                legendgroup='alert',
                plot_areas=True,
                showlegend=True,
            )

        if analysis_performance_alerts is not None:
            figure.add_alert(
                data=analysis_performance_metric,
                alerts=analysis_performance_alerts,
                indices=analysis_chunk_indices,
                start_dates=analysis_chunk_start_dates,
                end_dates=analysis_chunk_end_dates,
                name='Alert',
                legendgroup='alert',
                plot_areas=False,
                showlegend=False,
            )

        return figure
