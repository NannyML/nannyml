#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing CBPE estimation results and plotting implementations."""

import pandas as pd
from plotly import graph_objects as go

from nannyml import InvalidArgumentsException
from nannyml.base import AbstractEstimator, AbstractEstimatorResult
from nannyml.plots import CHUNK_KEY_COLUMN_NAME
from nannyml.plots._step_plot import _step_plot

SUPPORTED_METRIC_VALUES = ['roc_auc', 'f1', 'precision', 'recall', 'specificity', 'accuracy']


class CBPEPerformanceEstimatorResult(AbstractEstimatorResult):
    """Contains results for CBPE estimation and adds plotting functionality."""

    def __init__(self, results_data: pd.DataFrame, estimator: AbstractEstimator):
        super().__init__(results_data)

        from .cbpe import CBPE

        if not isinstance(estimator, CBPE):
            raise RuntimeError(
                f"{estimator.__class__.__name__} is not an instance of type " f"DataReconstructionDriftCalculator"
            )
        self.estimator = estimator

    @property
    def estimator_name(self) -> str:
        return 'confidence_based_performance_estimation'

    def plot(
        self, kind: str = 'performance', metric: str = None, plot_reference: bool = False, *args, **kwargs
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
        metric: str, default=None
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
            if metric not in SUPPORTED_METRIC_VALUES:
                raise InvalidArgumentsException(
                    f"unknown 'metric' value: '{metric}'. " f"Supported values are {SUPPORTED_METRIC_VALUES}."
                )
            return _plot_cbpe_performance_estimation(self.data, self.estimator, metric, plot_reference)
        else:
            raise InvalidArgumentsException(f"unknown plot kind '{kind}'. " f"Please provide on of: ['performance'].")

    # @property
    # def plots(self) -> Dict[str, go.Figure]:
    #     plots: Dict[str, go.Figure] = {}
    #     for metric in self.metrics:
    #         plots[f'estimated_{metric}'] = _plot_cbpe_performance_estimation(self.data, metric)
    #     return plots


def _plot_cbpe_performance_estimation(
    estimation_results: pd.DataFrame, estimator, metric: str, plot_reference: bool
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

    estimation_results['period'] = 'analysis'
    estimation_results['estimated'] = True

    if plot_reference:
        reference_results = estimator.previous_reference_results
        reference_results['period'] = 'reference'
        reference_results['estimated'] = False
        estimation_results = pd.concat([reference_results, estimation_results], ignore_index=True)

    # TODO: hack, assembling single results column to pass to plotting, overriding alert cols
    estimation_results['plottable'] = estimation_results.apply(
        lambda r: r[f'estimated_{metric}'] if r['period'] == 'analysis' else r[f'realized_{metric}'], axis=1
    )
    estimation_results['alert'] = estimation_results.apply(
        lambda r: r[f'alert_{metric}'] if r['period'] == 'analysis' else False, axis=1
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
        lower_threshold_column_name=f'lower_threshold_{metric}',
        upper_threshold_column_name=f'upper_threshold_{metric}',
        threshold_legend_label='Performance threshold',
        title=f'CBPE - Estimated {metric}',
        y_axis_title=f'{metric}',
        v_line_separating_analysis_period=plot_period_separator,
        estimated_column_name='estimated',
        lower_confidence_column_name=f'lower_confidence_{metric}',
        upper_confidence_column_name=f'upper_confidence_{metric}',
    )

    return fig
