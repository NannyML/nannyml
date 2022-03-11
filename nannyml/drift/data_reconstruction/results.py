#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Implementation of the Data Reconstruction Drift Calculator."""

import plotly.graph_objects as go

from nannyml.drift.base import DriftResult
from nannyml.plots import CHUNK_KEY_COLUMN_NAME
from nannyml.plots._line_plot import _line_plot


class DataReconstructionDriftCalculatorResult(DriftResult):
    """Contains the results of the data reconstruction drift calculation and adds functionality like plotting."""

    def plot(self, *args, **kwargs) -> go.Figure:
        """Renders a line plot of the ``reconstruction_error`` of the data reconstruction drift calculation results.

        Chunks are set on a time-based X-axis by using the period containing their observations.
        Chunks of different partitions (``reference`` and ``analysis``) are represented using different colors and
        a vertical separation if the drift results contain multiple partitions.


        Returns
        -------
        fig: plotly.graph_objects.Figure
            A ``Figure`` object containing the requested drift plot. Can be saved to disk or shown rendered on screen
            using ``fig.show()``.
        """
        plot_partition_separator = len(self.data.value_counts()) > 1
        self.data['thresholds'] = list(zip(self.data.lower_threshold, self.data.upper_threshold))
        fig = _line_plot(
            table=self.data,
            metric_column_name='reconstruction_error',
            chunk_column_name=CHUNK_KEY_COLUMN_NAME,
            drift_column_name='alert',
            threshold_column_name='thresholds',
            title='Data Reconstruction Drift',
            y_axis_title='Reconstruction Error',
            v_line_separating_analysis_period=plot_partition_separator,
        )

        return fig
