#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing the results of performance calculations and associated plots."""
from typing import Union

import pandas as pd
import plotly.graph_objects as go

from nannyml import InvalidArgumentsException
from nannyml.base import AbstractCalculator, AbstractCalculatorResult
from nannyml.plots import CHUNK_KEY_COLUMN_NAME
from nannyml.plots._step_plot import _step_plot

from .._typing import derive_use_case
from .metrics import Metric, MetricFactory


class PerformanceCalculatorResult(AbstractCalculatorResult):
    """Contains the results of performance calculation and adds plotting functionality."""

    def __init__(
        self,
        results_data: pd.DataFrame,
        calculator: AbstractCalculator,
    ):
        """Creates a new PerformanceCalculatorResult instance.

        Parameters
        ----------

        """
        super().__init__(results_data)

        from .calculator import PerformanceCalculator

        if not isinstance(calculator, PerformanceCalculator):
            raise RuntimeError(
                f"{calculator.__class__.__name__} is not an instance of type " f"UnivariateStatisticalDriftCalculator"
            )
        self.calculator = calculator

    @property
    def calculator_name(self) -> str:
        return "performance_calculator"

    def plot(
        self,
        kind: str = 'performance',
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
        metric: Union[str, Metric], default=None
            The name of the metric to plot. Value should be one of:
            - 'roc_auc'
            - 'f1'
            - 'precision'
            - 'recall'
            - 'specificity'
            - 'accuracy'
        plot_reference: bool, default=False
            Indicates whether to include the reference period in the plot or not. Defaults to ``False``.

        Examples
        --------
        >>> import nannyml.metadata.extraction
        >>> import nannyml as nml
        >>> ref_df, ana_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>> metadata = nannyml.metadata.extraction.extract_metadata(ref_df)
        >>> calculator = nml.PerformanceCalculator(model_metadata=metadata, chunk_period='W')
        >>> calculator.fit(ref_df)
        >>> realized_performance = calculator.calculate(ana_df)
        >>> # plot the calculated performance metrics
        >>> for m in calculator.metrics:
        >>>     realized_performance.plot(kind='performance', metric=m).show()
        """
        if kind == 'performance':
            if 'metric' not in kwargs:
                raise InvalidArgumentsException("missing value for parameter 'metric'")
            return _plot_performance_metric(self.data, self.calculator, plot_reference, kwargs['metric'])
        else:
            raise InvalidArgumentsException(f"unknown plot kind '{kind}'. " f"Please provide on of: ['performance'].")

    # @property
    # def plots(self) -> Dict[str, go.Figure]:
    #     return {metric: self.plot(kind='performance', metric=metric) for metric in self._metrics}


def _plot_performance_metric(
    results_data: pd.DataFrame, calculator, plot_reference: bool, metric: Union[str, Metric]
) -> go.Figure:
    """Renders a line plot of a selected metric of the performance calculation results.

    Chunks are set on a time-based X-axis by using the period containing their observations.
    Chunks of different periods (``reference`` and ``analysis``) are represented using different colors and
    a vertical separation if the drift results contain multiple periods.


    Parameters
    ----------
    results_data : pd.DataFrame
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
    results_data = results_data.copy()

    if isinstance(metric, str):
        metric = MetricFactory.create(metric, derive_use_case(calculator.y_pred_proba), {'calculator': calculator})

    plot_period_separator = plot_reference

    results_data['period'] = 'analysis'
    if plot_reference:
        reference_results = calculator.previous_reference_results
        reference_results['period'] = 'reference'
        results_data = pd.concat([reference_results, results_data], ignore_index=True)

    # Plot metric performance
    fig = _step_plot(
        table=results_data,
        metric_column_name=metric.column_name,
        chunk_column_name=CHUNK_KEY_COLUMN_NAME,
        drift_column_name=f'{metric.column_name}_alert',
        drift_legend_label='Degraded performance',
        hover_labels=['Chunk', metric.display_name, 'Target data'],
        hover_marker_labels=['Reference', 'No change', 'Change'],
        lower_threshold_column_name=f'{metric.column_name}_lower_threshold',
        upper_threshold_column_name=f'{metric.column_name}_upper_threshold',
        threshold_legend_label='Performance threshold',
        partial_target_column_name='targets_missing_rate',
        title=f'Realized performance: {metric.display_name}',
        y_axis_title='Realized performance',
        v_line_separating_analysis_period=plot_period_separator,
    )

    return fig
