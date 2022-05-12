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

SUPPORTED_METRIC_VALUES = ['roc_auc', 'f1', 'precision', 'recall', 'specificity', 'accuracy']


class CBPEPerformanceEstimatorResult(PerformanceEstimatorResult):
    """Contains results for CBPE estimation and adds plotting functionality."""

    def plot(self, kind: str = 'performance', metric: str = None, *args, **kwargs) -> go.Figure:
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
            The metric to plot when rendering a plot of kind 'performance'.

        Examples
        --------
        >>> import nannyml as nml
        >>> ref_df, ana_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>> metadata = nml.extract_metadata(ref_df, model_type=nml.ModelType.CLASSIFICATION_BINARY)
        >>> estimator = nml.CBPE(model_metadata=metadata, chunk_period='W')
        >>> estimator.fit(ref_df)
        >>> estimates = estimator.estimate(ana_df)
        >>> # plot the estimated performance
        >>> estimates.plot(kind='performance').show()

        """
        if kind == 'performance':
            if metric is None:
                raise InvalidArgumentsException(
                    "no value for 'metric' given. Please provide the name of a metric to display."
                )
            if metric not in SUPPORTED_METRIC_VALUES:
                raise InvalidArgumentsException(
                    f"unknown 'metric' value: '{metric}'. " f"Supported values are {SUPPORTED_METRIC_VALUES}."
                )
            return _plot_cbpe_performance_estimation(self.data, metric)
        else:
            raise InvalidArgumentsException(f"unknown plot kind '{kind}'. " f"Please provide on of: ['performance'].")


def _plot_cbpe_performance_estimation(estimation_results: pd.DataFrame, metric: str) -> go.Figure:
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
    metric: str, default=None
            The metric to plot when rendering a plot of kind 'performance'.

    Returns
    -------
    fig: plotly.graph_objects.Figure
        A ``Figure`` object containing the requested performance estimation plot.
        Can be saved to disk or shown rendered on screen using ``fig.show()``.
    """
    estimation_results = estimation_results.copy(deep=True)

    estimation_results['thresholds'] = list(
        zip(estimation_results[f'lower_threshold_{metric}'], estimation_results[f'upper_threshold_{metric}'])
    )

    estimation_results['estimated'] = estimation_results['partition'].apply(lambda r: r == 'analysis')

    plot_partition_separator = len(estimation_results['partition'].value_counts()) > 1

    # TODO: hack, assembling single results column to pass to plotting, overriding alert cols
    estimation_results['plottable'] = estimation_results.apply(
        lambda r: r[f'estimated_{metric}'] if r['partition'] == 'analysis' else r[f'realized_{metric}'], axis=1
    )
    estimation_results['alert'] = estimation_results.apply(
        lambda r: r[f'alert_{metric}'] if r['partition'] == 'analysis' else False, axis=1
    )

    # Plot estimated performance
    fig = _step_plot(
        table=estimation_results,
        metric_column_name='plottable',
        chunk_column_name=CHUNK_KEY_COLUMN_NAME,
        chunk_legend_labels=[f'Reference period (realized {metric})', f'Analysis period (estimated {metric})'],
        drift_column_name='alert',
        drift_legend_label='Degraded performance',
        hover_labels=['Chunk', f'{metric}', 'Target data'],
        hover_marker_labels=['Reference', 'No change', 'Change'],
        threshold_column_name='thresholds',
        threshold_legend_label='Performance threshold',
        title=f'CBPE - Estimated {metric}',
        y_axis_title=f'{metric}',
        v_line_separating_analysis_period=plot_partition_separator,
        estimated_column_name='estimated',
        confidence_column_name=f'confidence_{metric}',
    )

    return fig
