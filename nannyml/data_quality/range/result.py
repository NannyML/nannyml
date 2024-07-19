#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

"""Contains the results of the univariate statistical drift calculation and provides plotting functionality."""
from __future__ import annotations

import warnings
from typing import List, Optional

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import pandas as pd

import plotly.graph_objects as go

from nannyml._typing import Key
from nannyml.base import PerColumnResult
from nannyml.chunk import Chunker

# from nannyml.exceptions import InvalidArgumentsException
from nannyml.plots.blueprints.comparisons import ResultCompareMixin
from nannyml.plots.blueprints.metrics import plot_metrics
from nannyml.plots.components import Hover
from nannyml.usage_logging import UsageEvent, log_usage


class Result(PerColumnResult, ResultCompareMixin):
    """Contains the results of the univariate statistical drift calculation and provides plotting functionality."""

    def __init__(
        self,
        results_data: pd.DataFrame,
        column_names: List[str],
        data_quality_metric: str,
        timestamp_column_name: Optional[str],
        chunker: Chunker,
    ):
        super().__init__(results_data, column_names)

        self.timestamp_column_name = timestamp_column_name
        self.data_quality_metric = data_quality_metric
        self.chunker = chunker

    def keys(self) -> List[Key]:
        return [
            Key(
                properties=(column_name,),
                display_names=(column_name, f"{self.data_quality_metric.replace('_', ' ').title()}"),
            )
            for column_name in self.column_names
        ]

    @log_usage(UsageEvent.DQ_CALC_UNSEEN_VALUES_PLOT)
    def plot(
        self,
        *args,
        **kwargs,
    ) -> go.Figure:
        """

        Parameters
        ----------

        Returns
        -------
        fig: :class:`plotly.graph_objs._figure.Figure`
            A :class:`~plotly.graph_objs._figure.Figure` object containing the requested drift plot.

            Can be saved to disk using the :meth:`~plotly.graph_objs._figure.Figure.write_image` method
            or shown rendered on screen using the :meth:`~plotly.graph_objs._figure.Figure.show` method.

        Examples
        --------
        >>> import nannyml as nml
        >>> reference, analysis, _ = nml.load_synthetic_car_price_dataset()
        >>> column_names = [col for col in reference.columns if col not in [
        ...     'car_age', 'km_driven', 'price_new', 'accident_count', 'door_count','timestamp', 'y_pred', 'y_true']]
        >>> calc = nml.UnseenValuesCalculator(
        ...     column_names=column_names,
        ...     timestamp_column_name='timestamp',
        ... ).fit(reference)
        >>> res = calc.calculate(analysis)
        >>> res.filter(period='analysis').plot().show()

        """
        return plot_metrics(
            self,
            title='Data Quality ',
            hover=Hover(
                template='%{period} &nbsp; &nbsp; %{alert} <br />'
                'Chunk: <b>%{chunk_key}</b> &nbsp; &nbsp; %{x_coordinate} <br />'
                '%{metric_name}: <b>%{metric_value}</b><b r />',
                show_extra=True,
            ),
            subplot_title_format='{display_names[1]} for <b>{display_names[0]}</b>',
            subplot_y_axis_title_format='{display_names[1]}',
        )
