#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""The classes representing the results of a target distribution calculation."""
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from nannyml._typing import ProblemType
from nannyml.base import AbstractCalculator, AbstractCalculatorResult
from nannyml.exceptions import InvalidArgumentsException
from nannyml.plots._joy_plot import _joy_plot
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

    def plot(self, kind: str = 'target_drift', plot_reference: bool = False, *args, **kwargs) -> Optional[go.Figure]:
        """Renders plots for metrics returned by the target distribution calculator.

        You can render a step plot of the mean target distribution or the statistical tests per chunk.

        Select a plot using the ``kind`` parameter:

        - ``distribution``
                plots the drift metric per :class:`~nannyml.chunk.Chunk` for the model predictions ``y_pred``.

        Parameters
        ----------
        kind: str, default='distribution'
            The kind of plot to show. Allowed values are ``target_drift`` and ``target_distribution``.
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
        >>> results.plot(kind='target_drift', plot_reference=True).show()
        >>> results.plot(kind='target_distribution', plot_reference=True).show()
        """
        if kind == 'target_drift':
            return self._plot_target_drift(plot_reference)
        elif kind == 'target_distribution':
            return self._plot_target_distribution(plot_reference)
        else:
            raise InvalidArgumentsException(f"unknown plot kind '{kind}'. " f"Please provide one of: ['distribution'].")

    def _plot_target_drift(
        self,
        plot_reference: bool,
    ) -> go.Figure:
        """Renders a line plot of the drift metric for a given feature."""

        plot_period_separator = plot_reference
        data = self.data.copy()

        data['period'] = 'analysis'
        if plot_reference:
            reference_results = self.calculator.previous_reference_results
            if reference_results is None:
                raise RuntimeError(
                    f"could not plot distribution for '{self.calculator.y_true}': "
                    f"calculator is missing reference results\n{self.calculator}"
                )
            reference_results['period'] = 'reference'
            data = pd.concat([reference_results, data], ignore_index=True)

        if self.calculator.problem_type == ProblemType.REGRESSION:
            return _step_plot(
                table=data,
                metric_column_name='statistical_target_drift',
                chunk_column_name='key',
                drift_column_name='alert',
                hover_labels=['Chunk', 'KS statistic', 'Target data'],
                title=f'KS statistic over time for {self.calculator.y_true}',
                y_axis_title='KS statistic',
                v_line_separating_analysis_period=plot_period_separator,
            )
        elif self.calculator.problem_type in [ProblemType.CLASSIFICATION_BINARY, ProblemType.CLASSIFICATION_MULTICLASS]:
            return _step_plot(
                table=data,
                metric_column_name='metric_target_drift',
                chunk_column_name='key',
                drift_column_name='alert',
                hover_labels=['Chunk', 'Rate', 'Target data'],
                title=f'Target distribution over time for {self.calculator.y_true}',
                y_axis_title='Rate of positive occurrences',
                v_line_separating_analysis_period=plot_period_separator,
                partial_target_column_name='targets_missing_rate',
                statistically_significant_column_name='significant',
            )
        else:
            raise RuntimeError(
                "plot of kind 'target_drift' don't support " f"'{self.calculator.problem_type.value}' problems. "
            )

    def _plot_target_distribution(self, plot_reference: bool) -> go.Figure:
        plot_period_separator = plot_reference

        results_data = self.data

        results_data['period'] = 'analysis'

        if plot_reference:
            reference_results = self.calculator.previous_reference_results
            if reference_results is None:
                raise RuntimeError(
                    f"could not plot distribution for '{self.calculator.y_true}': "
                    f"calculator is missing reference results\n{self.calculator}"
                )
            reference_results['period'] = 'reference'
            results_data = pd.concat([reference_results, results_data.copy()], ignore_index=True)

        if self.calculator.problem_type in [ProblemType.CLASSIFICATION_BINARY, ProblemType.CLASSIFICATION_MULTICLASS]:
            return _step_plot(
                table=results_data,
                metric_column_name='statistical_target_drift',
                chunk_column_name='key',
                drift_column_name='alert',
                hover_labels=['Chunk', 'Chi-square statistic', 'Target data'],
                title=f'Chi-square statistic over time for {self.calculator.y_true} ',
                y_axis_title='Chi-square statistic',
                v_line_separating_analysis_period=plot_period_separator,
                partial_target_column_name='targets_missing_rate',
                statistically_significant_column_name='significant',
            )
        if self.calculator.problem_type == ProblemType.REGRESSION:
            feature_table = pd.concat(
                [
                    chunk.data.assign(key=chunk.key)
                    for chunk in self.calculator.chunker.split(
                        self.calculator.previous_analysis_data, self.calculator.timestamp_column_name
                    )
                ]
            )

            if plot_reference:
                reference_drift = self.calculator.previous_reference_results
                if reference_drift is None:
                    raise RuntimeError(
                        f"could not plot categorical distribution for target '{self.calculator.y_true}': "
                        f"calculator is missing reference results\n{self.calculator}"
                    )
                reference_drift['period'] = 'reference'
                results_data = pd.concat([reference_drift, results_data], ignore_index=True)

                reference_feature_table = pd.concat(
                    [
                        chunk.data.assign(key=chunk.key)
                        for chunk in self.calculator.chunker.split(
                            self.calculator.previous_reference_data, self.calculator.timestamp_column_name
                        )
                    ]
                )
                feature_table = pd.concat([reference_feature_table, feature_table], ignore_index=True)

            return _joy_plot(
                feature_table=feature_table,
                drift_table=results_data,
                chunk_column_name='key',
                drift_column_name='alert',
                feature_column_name=self.calculator.y_true,
                x_axis_title=f'{self.calculator.y_true}',
                post_kde_clip=None,
                title=f'Distribution over time for {self.calculator.y_true}',
                style='vertical',
            )
        else:
            raise RuntimeError(
                "plot of kind 'target_distribution' don't support " f"'{self.calculator.problem_type.value}' problems. "
            )
