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
from nannyml.plots import Figure
from nannyml.plots.blueprints.comparisons import plot_compare_performance_to_drift
from nannyml.plots.blueprints.metrics import plot_metric_list
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
        return plot_compare_performance_to_drift(
            performance_result=self.performance_result,
            drift_result=self.multivariate_result,
            performance_metric_display_name=self.performance_metric.display_name,
            performance_metric_column_name=self.performance_metric.column_name,
            drift_metric_display_name='Reconstruction error',
            drift_metric_column_name='reconstruction_error',
        )
