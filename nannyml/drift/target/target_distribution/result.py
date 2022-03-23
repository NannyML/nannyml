#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import pandas as pd
import plotly.graph_objects as go

from nannyml.exceptions import InvalidArgumentsException
from nannyml.metadata import ModelMetadata
from nannyml.plots import CHUNK_KEY_COLUMN_NAME
from nannyml.plots._step_plot import _step_plot


class TargetDistributionResult:
    def __init__(self, target_distribution: pd.DataFrame, model_metadata: ModelMetadata):
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
            return _plot_distribution(self.data, distribution, args, kwargs)
        else:
            raise InvalidArgumentsException(f"unknown plot kind '{kind}'. " f"Please provide one of: ['distribution'].")


def _plot_distribution(data: pd.DataFrame, distribution: str, *args, **kwargs) -> go.Figure:
    plot_partition_separator = len(data.value_counts()) > 1
    fig = _step_plot(
        table=data,
        metric_column_name='metric_target_drift' if distribution == 'metric' else 'statistical_target_drift',
        chunk_column_name=CHUNK_KEY_COLUMN_NAME,
        title=f'Target distribution ({distribution})',
        y_axis_title=f'Target distribution ({distribution})',
        v_line_separating_analysis_period=plot_partition_separator,
    )

    return fig
