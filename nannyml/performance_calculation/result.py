#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing the results of performance calculations and associated plots."""
from typing import Union

import pandas as pd
import plotly.graph_objects as go

from nannyml import InvalidArgumentsException
from nannyml.metadata import ModelMetadata
from nannyml.performance_calculation import Metric, MetricFactory
from nannyml.plots import CHUNK_KEY_COLUMN_NAME
from nannyml.plots._step_plot import _step_plot


class PerformanceCalculatorResult:
    """Contains the results of performance calculation and adds plotting functionality."""

    def __init__(
        self,
        performance_data: pd.DataFrame,
        model_metadata: ModelMetadata,
    ):
        """Creates a new PerformanceCalculatorResult instance.

        Parameters
        ----------
        performance_data : pd.DataFrame
            The results of the performance calculation.
        model_metadata :
            The metadata describing the monitored model.
        """
        self.data = performance_data
        self.metadata = model_metadata

    def plot(self, kind: str = 'performance', metric: Union[str, Metric] = None, *args, **kwargs) -> go.Figure:
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
        metric: str, default=None
            The name of the metric to plot. Value should be one of:
            - 'roc_auc'
            - 'f1'
            - 'precision'
            - 'recall'
            - 'specificity'
            - 'accuracy'

        """
        if kind == 'performance':
            return _plot_performance_metric(self.data, metric)
        else:
            raise InvalidArgumentsException(f"unknown plot kind '{kind}'. " f"Please provide on of: ['performance'].")


def _plot_performance_metric(
    performance_calculation_results: pd.DataFrame, metric: Union[str, Metric] = None
) -> go.Figure:
    """Renders a line plot of a selected metric of the performance calculation results.

    Chunks are set on a time-based X-axis by using the period containing their observations.
    Chunks of different partitions (``reference`` and ``analysis``) are represented using different colors and
    a vertical separation if the drift results contain multiple partitions.


    Parameters
    ----------
    performance_calculation_results : pd.DataFrame
        Results of the data CBPE performance estimation
    metric: str, default=None
            The name of the metric to plot. Value should be one of:
            - 'roc_auc'
            - 'f1'
            - 'precision'
            - 'recall'
            - 'sensitivity'
            - 'specificity'
            - 'accuracy'

    Returns
    -------
    fig: plotly.graph_objects.Figure
        A ``Figure`` object containing the requested performance estimation plot.
        Can be saved to disk or shown rendered on screen using ``fig.show()``.
    """
    performance_calculation_results = performance_calculation_results.copy(deep=True)

    plot_partition_separator = len(performance_calculation_results['partition'].value_counts()) > 1

    if isinstance(metric, Metric):
        metric_column_name = metric.column_name
        metric_display_name = metric.display_name
    else:
        _metric = MetricFactory.create(metric)  # type: ignore
        metric_column_name = _metric.column_name
        metric_display_name = _metric.display_name

    # Plot metric performance
    fig = _step_plot(
        table=performance_calculation_results,
        metric_column_name=metric_column_name,
        chunk_column_name=CHUNK_KEY_COLUMN_NAME,
        drift_column_name=f'{metric_column_name}_alert',
        drift_legend_label='Degraded performance',
        hover_labels=['Chunk', metric_display_name, 'Target data'],
        hover_marker_labels=['Reference', 'No change', 'Change'],
        threshold_column_name=f'{metric_column_name}_thresholds',
        threshold_legend_label='Performance threshold',
        partial_target_column_name='targets_missing_rate',
        title=f'Realized performance: {metric_display_name}',
        y_axis_title='Realized performance',
        v_line_separating_analysis_period=plot_partition_separator,
    )

    return fig
