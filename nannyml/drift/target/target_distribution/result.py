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
        """Renders a line plot of the target distribution.

        Chunks are set on a time-based X-axis by using the period containing their observations.
        Chunks of different periods (``reference`` and ``analysis``) are represented using different colors and
        a vertical separation if the drift results contain multiple periods.

        Parameters
        ----------
        kind: str
            The kind of plot to show. Restricted to the value 'distribution'.
        distribution: str, default='metric'
            The kind of distribution to plot. Restricted to the values 'metric' or 'statistical'.
        plot_reference: bool, default=False
            Indicates whether to include the reference period in the plot or not. Defaults to ``False``.

        Returns
        -------
        fig: plotly.graph_objects.Figure
            A ``Figure`` object containing the requested drift plot. Can be saved to disk or shown rendered on screen
            using ``fig.show()``.

        Examples
        --------
        >>> import nannyml as nml
        >>> ref_df, ana_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>> metadata = nml.extract_metadata(ref_df, model_type=nml.ModelType.CLASSIFICATION_BINARY)
        >>> target_distribution_calc = nml.TargetDistributionCalculator(model_metadata=metadata, chunk_period='W')
        >>> target_distribution_calc.fit(ref_df)
        >>> target_distribution = target_distribution_calc.calculate(ana_df)
        >>> # plot the distribution of the mean
        >>> target_distribution.plot(kind='metric').show()
        >>> # plot the Chi square statistic
        >>> target_distribution.plot(kind='statistical').show()
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
