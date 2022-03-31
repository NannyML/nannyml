#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""The classes representing the results of a target distribution calculation."""

import pandas as pd
import plotly.graph_objects as go

from nannyml.exceptions import InvalidArgumentsException
from nannyml.metadata import ModelMetadata
from nannyml.plots import CHUNK_KEY_COLUMN_NAME
from nannyml.plots._step_plot import _step_plot


class TargetDistributionResult:
    """Contains target distribution data and utilities to plot it."""

    def __init__(self, target_distribution: pd.DataFrame, model_metadata: ModelMetadata):
        """Creates a new instance of the TargetDistributionResults."""
        self.data = target_distribution
        self.metadata = model_metadata

    def plot(self, kind: str = 'distribution', distribution: str = 'metric', *args, **kwargs) -> go.Figure:
        """Renders a line plot of the target distribution.

        Chunks are set on a time-based X-axis by using the period containing their observations.
        Chunks of different partitions (``reference`` and ``analysis``) are represented using different colors and
        a vertical separation if the drift results contain multiple partitions.

        Parameters
        ----------
        kind: str
            The kind of plot to show. Restricted to the value 'distribution'.
        distribution: str, default='metric'
            The kind of distribution to plot. Restricted to the values 'metric' or 'statistical'

        Returns
        -------
        fig: plotly.graph_objects.Figure
            A ``Figure`` object containing the requested drift plot. Can be saved to disk or shown rendered on screen
            using ``fig.show()``.
        """
        if kind == 'distribution':
            return self._plot_distribution(distribution)
        else:
            raise InvalidArgumentsException(f"unknown plot kind '{kind}'. " f"Please provide one of: ['distribution'].")

    def _plot_distribution(self, distribution: str) -> go.Figure:
        plot_partition_separator = len(self.data.value_counts()) > 1

        if distribution == 'metric':
            fig = _step_plot(
                table=self.data,
                metric_column_name='metric_target_drift',
                chunk_column_name=CHUNK_KEY_COLUMN_NAME,
                drift_column_name='alert',
                hover_labels=['Chunk', 'Rate', 'Target data'],
                title=f'Target distribution over time for {self.metadata.target_column_name}',
                y_axis_title='Rate of positive occurrences',
                v_line_separating_analysis_period=plot_partition_separator,
                partial_target_column_name='targets_missing_rate',
                statistically_significant_column_name='significant',
            )
            return fig
        elif distribution == 'statistical':
            fig = _step_plot(
                table=self.data,
                metric_column_name='statistical_target_drift',
                chunk_column_name=CHUNK_KEY_COLUMN_NAME,
                drift_column_name='alert',
                hover_labels=['Chunk', 'Chi-square statistic', 'Target data'],
                title=f'Chi-square statistic over time for {self.metadata.target_column_name} ',
                y_axis_title='Chi-square statistic',
                v_line_separating_analysis_period=plot_partition_separator,
                partial_target_column_name='targets_missing_rate',
                statistically_significant_column_name='significant',
            )
            return fig
