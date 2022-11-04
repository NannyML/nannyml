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
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_estimation.confidence_based.metrics import Metric, MetricFactory
from nannyml.plots._step_plot import _step_plot

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

    def _filter(self, period: str, metrics: List[str] = None, *args, **kwargs) -> AbstractEstimatorResult:
        if metrics is None:
            metrics = [metric.column_name for metric in self.metrics]

        data = pd.concat([self.data.loc[:, (['chunk'])], self.data.loc[:, (metrics,)]], axis=1)
        if period != 'all':
            data = self.data.loc[self.data.loc[:, ('chunk', 'period')] == period, :]

        data = data.reset_index(drop=True)
        res = copy.deepcopy(self)
        res.data = data

        return res

    def plot(
        self,
        kind: str = 'performance',
        metric: Union[str, Metric] = None,
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


        Parameters
        ----------
        kind: str, default='performance'
            The kind of plot to render. Only the 'performance' plot is currently available.
        metric: Union[str, nannyml.performance_estimation.confidence_based.metrics.Metric]
            The metric to plot when rendering a plot of kind 'performance'.
        plot_reference: bool, default=False
            Indicates whether to include the reference period in the plot or not. Defaults to ``False``.

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
        >>> print(results.data)
                     key  start_index  ...  lower_threshold_roc_auc alert_roc_auc
        0       [0:4999]            0  ...                  0.97866         False
        1    [5000:9999]         5000  ...                  0.97866         False
        2  [10000:14999]        10000  ...                  0.97866         False
        3  [15000:19999]        15000  ...                  0.97866         False
        4  [20000:24999]        20000  ...                  0.97866         False
        5  [25000:29999]        25000  ...                  0.97866          True
        6  [30000:34999]        30000  ...                  0.97866          True
        7  [35000:39999]        35000  ...                  0.97866          True
        8  [40000:44999]        40000  ...                  0.97866          True
        9  [45000:49999]        45000  ...                  0.97866          True
        >>> for metric in estimator.metrics:
        >>>     results.plot(metric=metric, plot_reference=True).show()
        """
        if kind == 'performance':
            if metric is None:
                raise InvalidArgumentsException(
                    "no value for 'metric' given. Please provide the name of a metric to display."
                )
            if isinstance(metric, str):
                metric = MetricFactory.create(
                    metric,
                    self.problem_type,
                    y_pred_proba=self.y_pred_proba,
                    y_pred=self.y_pred,
                    y_true=self.y_true,
                    chunker=self.chunker,
                    timestamp_column_name=self.timestamp_column_name,
                )
            return self._plot_cbpe_performance_estimation(self.to_df(multilevel=False), self, metric, plot_reference)
        else:
            raise InvalidArgumentsException(f"unknown plot kind '{kind}'. " f"Please provide on of: ['performance'].")

    def _plot_cbpe_performance_estimation(
        self, estimation_results: pd.DataFrame, estimator, metric: Metric, plot_reference: bool
    ) -> go.Figure:
        """Renders a line plot of the ``reconstruction_error`` of the data reconstruction drift calculation results.

        Chunks are set on a time-based X-axis by using the period containing their observations.
        Chunks of different periods (``reference`` and ``analysis``) are represented using different colors and
        a vertical separation if the drift results contain multiple periods.

        If the ``realized_performance`` data is also provided, an extra line shall be plotted to allow an easy
        comparison of the estimated versus realized performance.

        Parameters
        ----------
        estimation_results : pd.DataFrame
            Results of the data CBPE performance estimation
        metric: str, default=None
                The metric to plot when rendering a plot of kind 'performance'.

        Returns
        -------
        fig: plotly.graph_objects.Figure
            A ``Figure`` object containing the requested performance estimation plot.
            Can be saved to disk or shown rendered on screen using ``fig.show()``.
        """
        estimation_results = estimation_results.copy()

        plot_period_separator = plot_reference

        estimation_results['estimated'] = True

        if not plot_reference:
            estimation_results = estimation_results[estimation_results['chunk_period'] == 'analysis']

        # TODO: hack, assembling single results column to pass to plotting, overriding alert cols
        estimation_results['plottable'] = estimation_results.apply(
            lambda r: r[f'{metric.column_name}_value']
            if r['chunk_period'] == 'analysis'
            else r[f'{metric.column_name}_realized'],
            axis=1,
        )
        estimation_results['alert'] = estimation_results.apply(
            lambda r: r[f'{metric.column_name}_alert'] if r['chunk_period'] == 'analysis' else False, axis=1
        )

        is_time_based_x_axis = self.timestamp_column_name is not None

        # Plot estimated performance
        fig = _step_plot(
            table=estimation_results,
            metric_column_name='plottable',
            chunk_column_name='chunk_key',
            chunk_type_column_name='chunk_period',
            chunk_index_column_name='chunk_index',
            chunk_legend_labels=[
                f'Reference period (realized {metric.display_name})',
                f'Analysis period (estimated {metric.display_name})',
            ],
            drift_column_name='alert',
            drift_legend_label='Degraded performance',
            hover_labels=['Chunk', f'{metric.display_name}', 'Target data'],
            hover_marker_labels=['Reference', 'No change', 'Change'],
            lower_threshold_column_name=f'{metric.column_name}_lower_threshold',
            upper_threshold_column_name=f'{metric.column_name}_upper_threshold',
            threshold_legend_label='Performance threshold',
            title=f'CBPE - Estimated {metric.display_name}',
            y_axis_title=f'{metric.display_name}',
            v_line_separating_analysis_period=plot_period_separator,
            estimated_column_name='estimated',
            lower_confidence_column_name=f'{metric.column_name}_lower_confidence_boundary',
            upper_confidence_column_name=f'{metric.column_name}_upper_confidence_boundary',
            sampling_error_column_name=f'{metric.column_name}_sampling_error',
            start_date_column_name='chunk_start_date' if is_time_based_x_axis else None,
            end_date_column_name='chunk_end_date' if is_time_based_x_axis else None,
        )

        return fig
