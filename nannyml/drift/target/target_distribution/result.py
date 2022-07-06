#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""The classes representing the results of a target distribution calculation."""

import pandas as pd
import plotly.graph_objects as go

from nannyml.base import AbstractCalculator, AbstractCalculatorResult
from nannyml.exceptions import InvalidArgumentsException
from nannyml.plots._step_plot import _step_plot


class TargetDistributionResult(AbstractCalculatorResult):
    """Contains target distribution data and utilities to plot it."""

    def __init__(self, results_data: pd.DataFrame, calculator: AbstractCalculator):
        """Creates a new instance of the TargetDistributionResults."""
        super().__init__(results_data)

        from .calculator import TargetDistributionCalculator

        if not isinstance(calculator, TargetDistributionCalculator):
            raise RuntimeError(
                f"{calculator.__class__.__name__} is not an instance of type " f"DataReconstructionDriftCalculator"
            )
        self.calculator = calculator

    @property
    def calculator_name(self) -> str:
        return 'target_distribution'

    def plot(
        self, kind: str = 'distribution', distribution: str = 'metric', plot_reference: bool = False, *args, **kwargs
    ) -> go.Figure:
        """Renders plots for metrics returned by the target distribution calculator.

        You can render a step plot of the mean target distribution or the statistical tests per chunk.

        Select a plot using the ``kind`` parameter:

        - ``distribution``
                plots the drift metric per :class:`~nannyml.chunk.Chunk` for the model predictions ``y_pred``.

        Parameters
        ----------
        kind: str, default='distribution'
            The kind of plot to show. Allowed values are ``distribution``.
        distribution: str, default='metric'
            The kind of distribution to plot. Allowed values are ``metric`` and ``statistical``.
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
        >>> calc = nml.TargetDistributionCalculator(
        >>>     y_true='work_home_actual',
        >>>     timestamp_column_name='timestamp'
        >>> )
        >>> calc.fit(reference_df)
        >>> results = calc.calculate(analysis_df.merge(target_df, on='identifier'))
        >>> print(results.data)  # check the numbers
                     key  start_index  end_index  ... thresholds  alert significant
        0       [0:4999]            0       4999  ...       0.05   True        True
        1    [5000:9999]         5000       9999  ...       0.05  False       False
        2  [10000:14999]        10000      14999  ...       0.05  False       False
        3  [15000:19999]        15000      19999  ...       0.05  False       False
        4  [20000:24999]        20000      24999  ...       0.05  False       False
        5  [25000:29999]        25000      29999  ...       0.05  False       False
        6  [30000:34999]        30000      34999  ...       0.05  False       False
        7  [35000:39999]        35000      39999  ...       0.05  False       False
        8  [40000:44999]        40000      44999  ...       0.05  False       False
        9  [45000:49999]        45000      49999  ...       0.05  False       False
        >>>
        >>> results.plot(distribution='metric', plot_reference=True).show()
        >>> results.plot(distribution='statistical', plot_reference=True).show()
        """
        if kind == 'distribution':
            return _plot_distribution(self.data, self.calculator, distribution, plot_reference)
        else:
            raise InvalidArgumentsException(f"unknown plot kind '{kind}'. " f"Please provide one of: ['distribution'].")

    # @property
    # def plots(self) -> Dict[str, go.Figure]:
    #     return {
    #         f'{self.metadata.target_column_name}_distribution_metric': self._plot_distribution('metric'),
    #         f'{self.metadata.target_column_name}_distribution_statistical': self._plot_distribution('statistical'),
    #     }


def _plot_distribution(data: pd.DataFrame, calculator, distribution: str, plot_reference: bool) -> go.Figure:
    plot_period_separator = plot_reference

    data['period'] = 'analysis'

    if plot_reference:
        reference_results = calculator.previous_reference_results
        reference_results['period'] = 'reference'
        data = pd.concat([reference_results, data.copy()], ignore_index=True)

    if distribution == 'metric':
        fig = _step_plot(
            table=data,
            metric_column_name='metric_target_drift',
            chunk_column_name='key',
            drift_column_name='alert',
            hover_labels=['Chunk', 'Rate', 'Target data'],
            title=f'Target distribution over time for {calculator.y_true}',
            y_axis_title='Rate of positive occurrences',
            v_line_separating_analysis_period=plot_period_separator,
            partial_target_column_name='targets_missing_rate',
            statistically_significant_column_name='significant',
        )
        return fig
    elif distribution == 'statistical':
        fig = _step_plot(
            table=data,
            metric_column_name='statistical_target_drift',
            chunk_column_name='key',
            drift_column_name='alert',
            hover_labels=['Chunk', 'Chi-square statistic', 'Target data'],
            title=f'Chi-square statistic over time for {calculator.y_true} ',
            y_axis_title='Chi-square statistic',
            v_line_separating_analysis_period=plot_period_separator,
            partial_target_column_name='targets_missing_rate',
            statistically_significant_column_name='significant',
        )
        return fig
