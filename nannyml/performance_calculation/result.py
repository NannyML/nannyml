#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Contains the results of the realized performance calculation and provides plotting functionality."""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Union

import pandas as pd
import plotly.graph_objects as go

from nannyml._typing import Key, ProblemType
from nannyml._typing import Result as ResultType
from nannyml.base import Abstract1DResult
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_calculation.metrics.base import Metric
from nannyml.plots.blueprints.comparisons import ResultCompareMixin
from nannyml.plots.blueprints.metrics import plot_metrics
from nannyml.usage_logging import UsageEvent, log_usage


class Result(Abstract1DResult, ResultCompareMixin):
    """Contains the results of the realized performance calculation and provides plotting functionality."""

    def __init__(
        self,
        results_data: pd.DataFrame,
        problem_type: ProblemType,
        y_pred: str,
        y_pred_proba: Optional[Union[str, Dict[str, str]]],
        y_true: str,
        metrics: List[Metric],
        timestamp_column_name: Optional[str] = None,
        reference_data: Optional[pd.DataFrame] = None,
        analysis_data: Optional[pd.DataFrame] = None,
    ):
        """Creates a new Result instance."""
        super().__init__(results_data)

        self.problem_type = problem_type

        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.y_pred = y_pred
        self.timestamp_column_name = timestamp_column_name
        self.metrics = metrics

        self.reference_data = reference_data
        self.analysis_data = analysis_data

    def _filter(self, period: str, metrics: Optional[List[str]] = None, *args, **kwargs) -> ResultType:
        if metrics is None:
            metrics = [metric.column_name for metric in self.metrics]

        data = pd.concat([self.data.loc[:, (['chunk'])], self.data.loc[:, (metrics,)]], axis=1)

        if period != 'all':
            data = data.loc[self.data.loc[:, ('chunk', 'period')] == period, :]

        data = data.reset_index(drop=True)

        res = copy.deepcopy(self)
        res.data = data
        res.metrics = [metric for metric in self.metrics if metric.column_name in metrics]

        return res

    def keys(self) -> List[Key]:
        return [
            Key(properties=(metric.column_name,), display_names=(f'realized {metric.display_name}',))
            for metric in self.metrics
        ]

    @log_usage(UsageEvent.PERFORMANCE_PLOT, metadata_from_kwargs=['kind'])
    def plot(
        self,
        kind: str = 'performance',
        *args,
        **kwargs,
    ) -> Optional[go.Figure]:
        """Render realized performance metrics.

            The following kinds of plots are available:

        - ``performance``
                a step plot showing the realized performance metric per :class:`~nannyml.chunk.Chunk` for
                a given metric.

        Parameters
        ----------
        kind: str, default='performance'
            The kind of plot to render. Only the 'performance' plot is currently available.

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
        >>> calc = nml.PerformanceCalculator(y_true='work_home_actual', y_pred='y_pred', y_pred_proba='y_pred_proba',
        >>>                                  timestamp_column_name='timestamp', metrics=['f1', 'roc_auc'])
        >>>
        >>> calc.fit(reference_df)
        >>>
        >>> results = calc.calculate(analysis_df.merge(target_df, on='identifier'))
        >>> print(results.data)
                     key  start_index  ...  roc_auc_upper_threshold roc_auc_alert
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
        >>> for metric in calc.metrics:
        >>>     results.plot(metric=metric, plot_reference=True).show()
        """
        if kind == 'performance':
            return plot_metrics(
                result=self,
                title='Realized performance',
            )
        else:
            raise InvalidArgumentsException(f"unknown plot kind '{kind}'. " f"Please provide on of: ['performance'].")
