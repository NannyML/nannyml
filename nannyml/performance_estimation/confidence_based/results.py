#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing CBPE estimation results and plotting implementations."""

import pandas as pd
from plotly import graph_objects as go

from nannyml import InvalidArgumentsException
from nannyml.performance_estimation.base import PerformanceEstimatorResult
from nannyml.plots import CHUNK_KEY_COLUMN_NAME
from nannyml.plots._step_plot import _step_plot


class CBPEPerformanceEstimatorResult(PerformanceEstimatorResult):
    """Contains results for CBPE estimation and adds plotting functionality."""

    def plot(self, kind: str = 'performance', *args, **kwargs) -> go.Figure:
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

        """
        if kind == 'performance':
            return _plot_cbpe_performance_estimation(self.data)
        else:
            raise InvalidArgumentsException(f"unknown plot kind '{kind}'. " f"Please provide on of: ['performance'].")


def _plot_cbpe_performance_estimation(estimation_results: pd.DataFrame) -> go.Figure:
    """Renders a line plot of the ``reconstruction_error`` of the data reconstruction drift calculation results.

    Chunks are set on a time-based X-axis by using the period containing their observations.
    Chunks of different partitions (``reference`` and ``analysis``) are represented using different colors and
    a vertical separation if the drift results contain multiple partitions.

    If the ``realized_performance`` data is also provided, an extra line shall be plotted to allow an easy
    comparison of the estimated versus realized performance.

    Parameters
    ----------
    estimation_results : pd.DataFrame
        Results of the data CBPE performance estimation

    Returns
    -------
    fig: plotly.graph_objects.Figure
        A ``Figure`` object containing the requested performance estimation plot.
        Can be saved to disk or shown rendered on screen using ``fig.show()``.
    """
    estimation_results = estimation_results.copy(deep=True)

    estimation_results['thresholds'] = list(zip(estimation_results.lower_threshold, estimation_results.upper_threshold))

    estimation_results['estimated'] = estimation_results['partition'].apply(lambda r: r == 'analysis')

    plot_partition_separator = len(estimation_results['partition'].value_counts()) > 1

    # TODO: hack, assembling single results column to pass to plotting, overriding alert cols
    estimation_results['plottable'] = estimation_results.apply(
        lambda r: r['estimated_roc_auc'] if r['partition'] == 'analysis' else r['realized_roc_auc'], axis=1
    )
    estimation_results['alert'] = estimation_results.apply(
        lambda r: r['alert'] if r['partition'] == 'analysis' else False, axis=1
    )

    # Plot estimated performance
    fig = _step_plot(
        table=estimation_results,
        metric_column_name='plottable',
        chunk_column_name=CHUNK_KEY_COLUMN_NAME,
        drift_column_name='alert',
        drift_legend_label='Degraded performance',
        hover_labels=['Chunk', 'Estimated ROC AUC', 'Target data'],
        hover_marker_labels=['Reference', 'No change', 'Change'],
        threshold_column_name='thresholds',
        threshold_legend_label='Performance threshold',
        title='CBPE - Estimated ROC AUC',
        y_axis_title='Estimated ROC AUC',
        v_line_separating_analysis_period=plot_partition_separator,
        estimated_column_name='estimated',
        confidence_column_name='confidence',
    )

    return fig
