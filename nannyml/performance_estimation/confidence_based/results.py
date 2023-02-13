#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing CBPE estimation results and plotting implementations."""
from __future__ import annotations
import copy
from typing import List, Optional, cast

import pandas as pd
from plotly import graph_objects as go

from nannyml._typing import Key, ModelOutputsType, ProblemType
from nannyml.base import Abstract1DResult
from nannyml.chunk import Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_estimation.confidence_based.metrics import Metric
from nannyml.plots.blueprints.comparisons import ResultCompareMixin
from nannyml.plots.blueprints.metrics import plot_metrics
from nannyml.usage_logging import UsageEvent, log_usage

SUPPORTED_METRIC_VALUES = [
    'roc_auc',
    'f1',
    'precision',
    'recall',
    'specificity',
    'accuracy',
    'confusion_matrix',
    'true_positive',
    'true_negative',
    'false_positive',
    'false_negative',
]


class Result(Abstract1DResult, ResultCompareMixin):
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
        super().__init__(results_data, metrics)

        # Be more specific about the metric type than the base class
        self.metrics: List[Metric]

        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.y_true = y_true
        self.timestamp_column_name = timestamp_column_name
        self.problem_type = problem_type
        self.chunker = chunker

    def _filter(self, period: str, metrics: Optional[List[str]] = None, *args, **kwargs) -> Result:
        """Filter the results based on the specified period and metrics.

        This function begins by expanding the metrics to all the metrics that were specified
        or if no metrics were specified, all the metrics that were used to calculate the results.
        Since some metrics have multiple components, we expand these to their individual components.
        For example, the ``confusion_matrix`` metric has four components: ``true_positive``,
        ``true_negative``, ``false_positive``, and ``false_negative``.  Specifying ``confusion_matrix``
        or, for example, ``true_positive`` are both valid. We then filter the results based on the
        specified period and metrics.
        """
        if metrics is None:
            filtered_metrics = self.metrics
        else:
            filtered_metrics = []
            for name in metrics:
                if name not in SUPPORTED_METRIC_VALUES:
                    raise InvalidArgumentsException(
                        f"invalid metric '{name}'. Please choose from {SUPPORTED_METRIC_VALUES}"
                    )

                m = self._get_metric_by_name(name)

                if m:
                    filtered_metrics = filtered_metrics + [m]
                else:
                    raise InvalidArgumentsException(f"no '{name}' in result, did you calculate it?")

        metric_column_names = [name for metric in filtered_metrics for name in metric.column_names]

        data = pd.concat([self.data.loc[:, (['chunk'])], self.data.loc[:, (metric_column_names,)]], axis=1)
        if period != 'all':
            data = data.loc[data.loc[:, ('chunk', 'period')] == period, :]

        data = data.reset_index(drop=True)
        res = copy.deepcopy(self)
        res.data = data
        res.metrics = filtered_metrics

        return res

    def _get_metric_by_name(self, name: str) -> Optional[Metric]:
        for metric in self.metrics:
            # If we match the metric by name, return the metric
            # E.g. matching the name 'confusion_matrix'
            if name == metric.name:
                return metric
            # If we match one of the metric component names
            # E.g. matching the name 'true_positive' with the confusion matrix metric
            elif name in metric.column_names:
                # Only retain the component whose column name was given to filter on
                res = copy.deepcopy(metric)
                res.components = list(filter(lambda c: c[1] == name, metric.components))
                return res
            else:
                continue
        return None

    def keys(self) -> List[Key]:
        return [
            Key(
                properties=(component[1],),
                display_names=(
                    f'estimated {component[0]}',
                    component[0],
                ),
            )
            for metric in self.metrics
            for component in cast(Metric, metric).components
        ]

    @log_usage(UsageEvent.CBPE_PLOT, metadata_from_kwargs=['kind'])
    def plot(
        self,
        kind: str = 'performance',
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
            return plot_metrics(
                self,
                title='Estimated performance <b>(CBPE)</b>',
                subplot_title_format='Estimated <b>{display_names[1]}</b>',
                subplot_y_axis_title_format='{display_names[1]}',
            )
        else:
            raise InvalidArgumentsException(f"unknown plot kind '{kind}'. " f"Please provide on of: ['performance'].")
