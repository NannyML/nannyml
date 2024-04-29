#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Contains the results of the realized performance calculation and provides filtering and plotting functionality."""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Union, cast

import pandas as pd
import plotly.graph_objects as go

from nannyml._typing import Key, ProblemType, Self
from nannyml.base import PerMetricResult
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_calculation import SUPPORTED_METRIC_FILTER_VALUES
from nannyml.performance_calculation.metrics.base import Metric
from nannyml.plots.blueprints.comparisons import ResultCompareMixin
from nannyml.plots.blueprints.metrics import plot_metrics
from nannyml.usage_logging import UsageEvent, log_usage


class Result(PerMetricResult[Metric], ResultCompareMixin):
    """Wraps performance calculation results and provides filtering and plotting functionality."""

    metrics: List[Metric]

    def __init__(
        self,
        results_data: pd.DataFrame,
        problem_type: ProblemType,
        y_pred: Optional[str],
        y_pred_proba: Optional[Union[str, Dict[str, str]]],
        y_true: str,
        metrics: List[Metric],
        timestamp_column_name: Optional[str] = None,
        reference_data: Optional[pd.DataFrame] = None,
        analysis_data: Optional[pd.DataFrame] = None,
    ):
        """Creates a new Result instance.

        Parameters
        ----------
        results_data: pd.DataFrame
            Results data returned by a CBPE estimator.
        problem_type: ProblemType
            Determines which method to use. Allowed values are:

                - 'regression'
                - 'classification_binary'
                - 'classification_multiclass'
        y_pred: str
            The name of the column containing your model predictions.
        y_pred_proba: Union[str, Dict[str, str]]
            Name(s) of the column(s) containing your model output.

                - For binary classification, pass a single string refering to the model output column.
                - For multiclass classification, pass a dictionary that maps a class string to the column name \
                containing model outputs for that class.
        y_true: str
            The name of the column containing target values (that are provided in reference data during fitting).
        metrics: List[nannyml.performance_calculation.metrics.base.Metric]
            List of metrics to evaluate.
        timestamp_column_name: str, default=None
            The name of the column containing the timestamp of the model prediction.
            If not given, plots will not use a time-based x-axis but will use the index of the chunks instead.
        reference_data: pd.DataFrame, default=None
            The reference data used for fitting. Must have target data available.
        analysis_data: pd.DataFrame, default=None
            The data on which NannyML calculates the perfomance.

        """
        super().__init__(results_data, metrics)

        self.problem_type = problem_type

        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.y_pred = y_pred
        self.timestamp_column_name = timestamp_column_name

        self.reference_data = reference_data
        self.analysis_data = analysis_data

    def keys(self) -> List[Key]:
        """Creates a list of keys where each Key is a `namedtuple('Key', 'properties display_names')`."""
        return [
            Key(
                properties=(component[1],),
                display_names=(
                    f'realized {component[0]}',
                    component[0],
                    metric.name,
                ),
            )
            for metric in self.metrics
            for component in cast(Metric, metric).components
        ]

    @log_usage(UsageEvent.PERFORMANCE_PLOT, metadata_from_kwargs=['kind'])
    def plot(
        self,
        kind: str = 'performance',
        *args,
        **kwargs,
    ) -> go.Figure:
        """Render realized performance metrics.

        This function will return a :class:`plotly.graph_objects.Figure` object.

        Parameters
        ----------
        kind: str, default='performance'
            The kind of plot to render. Only the 'performance' plot is currently available.

        Raises
        ------
        InvalidArgumentsException: when an unknown plot ``kind`` is provided.

        Returns
        -------
        fig: :class:`plotly.graph_objs._figure.Figure`
            A :class:`~plotly.graph_objs._figure.Figure` object containing the requested drift plot.

            Can be saved to disk using the :meth:`~plotly.graph_objs._figure.Figure.write_image` method
            or shown rendered on screen using the :meth:`~plotly.graph_objs._figure.Figure.show` method.

        Examples
        --------
        >>> import nannyml as nml
        >>> from IPython.display import display
        >>> reference_df, analysis_df, analysis_targets_df = nml.load_synthetic_car_loan_dataset()
        >>> analysis_df = analysis_df.merge(analysis_targets_df, left_index=True, right_index=True)
        >>> display(reference_df.head(3))
        >>> calc = nml.PerformanceCalculator(
        ...     y_pred_proba='y_pred_proba',
        ...     y_pred='y_pred',
        ...     y_true='repaid',
        ...     timestamp_column_name='timestamp',
        ...     problem_type='classification_binary',
        ...     metrics=['roc_auc', 'f1', 'precision', 'recall', 'specificity', 'accuracy'],
        ...     chunk_size=5000)
        >>> calc.fit(reference_df)
        >>> results = calc.calculate(analysis_df)
        >>> display(results.filter(period='analysis').to_df())
        >>> display(results.filter(period='reference').to_df())
        >>> figure = results.plot()
        >>> figure.show()
        """
        if kind == 'performance':
            return plot_metrics(
                result=self,
                title='Realized performance',
                subplot_title_format='Realized <b>{display_names[1]}</b>',
                subplot_y_axis_title_format='{display_names[1]}',
            )
        else:
            raise InvalidArgumentsException(f"unknown plot kind '{kind}'. " f"Please provide on of: ['performance'].")

    def _filter(self, period: str, metrics: Optional[List[str]] = None, *args, **kwargs) -> Self:
        """Filter the results based on the specified period and metrics."""
        if metrics is None:
            filtered_metrics = self.metrics
        else:
            filtered_metrics = []
            for name in metrics:
                if name not in SUPPORTED_METRIC_FILTER_VALUES:
                    raise InvalidArgumentsException(f"invalid metric '{name}'")

                m = self._get_metric_by_name(name)

                if m:
                    filtered_metrics = filtered_metrics + [m]
                else:
                    raise InvalidArgumentsException(f"no '{name}' in result, did you calculate it?")

        metric_column_names = [name for metric in filtered_metrics for name in metric.column_names]

        res = super()._filter(period, metric_column_names, *args, **kwargs)
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
