#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Implementation of the Data Reconstruction Drift Calculator."""

import pandas as pd
import plotly.graph_objects as go

from nannyml.drift.base import DriftResult
from nannyml.exceptions import InvalidArgumentsException
from nannyml.plots import CHUNK_KEY_COLUMN_NAME
from nannyml.plots._step_plot import _step_plot


class DataReconstructionDriftCalculatorResult(DriftResult):
    """Contains the results of the data reconstruction drift calculation and adds functionality like plotting."""

    def plot(self, kind: str = 'drift', *args, **kwargs) -> go.Figure:
        """Renders a line plot of the ``reconstruction_error`` of the data reconstruction drift calculation results.

        Chunks are set on a time-based X-axis by using the period containing their observations.
        Chunks of different partitions (``reference`` and ``analysis``) are represented using different colors and
        a vertical separation if the drift results contain multiple partitions.

        The different plot kinds that are available:

        - ``drift``: plots drift per :class:`~nannyml.chunk.Chunk` for a chunked data set.

        Returns
        -------
        fig: plotly.graph_objects.Figure
            A ``Figure`` object containing the requested drift plot. Can be saved to disk or shown rendered on screen
            using ``fig.show()``.
        """
        if kind == 'drift':
            return _plot_drift(self.data, args, kwargs)
        else:
            raise InvalidArgumentsException(
                f"unknown plot kind '{kind}'. "
                f"Please provide on of: ['feature_drift', 'feature_distribution', "
                f"'prediction_drift', 'prediction_distribution']."
            )


def _plot_drift(data: pd.DataFrame, *args, **kwargs) -> go.Figure:
    plot_partition_separator = len(data.value_counts()) > 1
    data['thresholds'] = list(zip(data.lower_threshold, data.upper_threshold))
    fig = _step_plot(
        table=data,
        metric_column_name='reconstruction_error',
        chunk_column_name=CHUNK_KEY_COLUMN_NAME,
        drift_column_name='alert',
        threshold_column_name='thresholds',
        hover_labels=['Chunk', 'Reconstruction error', 'Target data'],
        title='Data Reconstruction Drift',
        y_axis_title='Reconstruction Error',
        v_line_separating_analysis_period=plot_partition_separator,
    )

    return fig
