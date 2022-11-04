import copy
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from plotly.graph_objects import Figure

from nannyml import Chunker
from nannyml._typing import ProblemType
from nannyml.base import AbstractEstimatorResult
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_estimation.direct_loss_estimation.metrics import Metric, MetricFactory
from nannyml.plots._step_plot import _step_plot


class Result(AbstractEstimatorResult):
    def __init__(
        self,
        results_data: pd.DataFrame,
        metrics: List[Metric],
        feature_column_names: List[str],
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        tune_hyperparameters: bool,
        hyperparameter_tuning_config: Dict[str, Any],
        hyperparameters: Optional[Dict[str, Any]],
        timestamp_column_name: Optional[str] = None,
    ):
        super().__init__(results_data)

        self.metrics = metrics
        self.feature_column_names = feature_column_names
        self.y_pred = y_pred
        self.y_true = y_true
        self.timestamp_column_name = timestamp_column_name

        self.chunker = chunker

        self.tune_hyperparameters = tune_hyperparameters
        self.hyperparameter_tuning_config = (hyperparameter_tuning_config,)
        self.hyperparameters = hyperparameters

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
    ) -> Figure:
        if kind == 'performance':
            if metric is None:
                raise InvalidArgumentsException(
                    "no value for 'metric' given. Please provide the name of a metric to display."
                )
            if isinstance(metric, str):
                metric = MetricFactory.create(
                    metric,
                    ProblemType.REGRESSION,
                    feature_column_names=self.feature_column_names,
                    y_true=self.y_true,
                    y_pred=self.y_pred,
                    chunker=self.chunker,
                    tune_hyperparameters=self.tune_hyperparameters,
                    hyperparameter_tuning_config=self.hyperparameter_tuning_config,
                    hyperparameters=self.hyperparameters,
                )

            return self._plot_direct_error_estimation_performance(metric, plot_reference)
        else:
            raise InvalidArgumentsException(f"unknown plot kind '{kind}'. " f"Please provide on of: ['performance'].")

    def _plot_direct_error_estimation_performance(self, metric: Metric, plot_reference: bool) -> Figure:
        estimation_results = self.to_df(multilevel=False)

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
        estimation_results[f'{metric.column_name}_alert'] = estimation_results.apply(
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
            start_date_column_name='chunk_start_date' if is_time_based_x_axis else None,
            end_date_column_name='chunk_end_date' if is_time_based_x_axis else None,
            chunk_legend_labels=[
                f'Reference period (realized {metric.display_name})',
                f'Analysis period (estimated {metric.display_name})',
            ],
            drift_column_name=f'{metric.column_name}_alert',
            drift_legend_label='Degraded performance',
            hover_labels=['Chunk', f'{metric.display_name}', 'Target data'],
            hover_marker_labels=['Reference', 'No change', 'Change'],
            lower_threshold_column_name=f'{metric.column_name}_lower_threshold',
            upper_threshold_column_name=f'{metric.column_name}_upper_threshold',
            threshold_legend_label='Performance threshold',
            title=f'DLE - Estimated {metric.display_name}',
            y_axis_title=f'{metric.display_name}',
            v_line_separating_analysis_period=plot_period_separator,
            estimated_column_name='estimated',
            lower_confidence_column_name=f'{metric.column_name}_lower_confidence_boundary',
            upper_confidence_column_name=f'{metric.column_name}_upper_confidence_boundary',
            sampling_error_column_name=f'{metric.column_name}_sampling_error',
        )

        return fig
