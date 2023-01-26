#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Contains the results of the data reconstruction drift calculation and provides plotting functionality."""
from __future__ import annotations

import copy
from collections import namedtuple
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go

from nannyml._typing import Key
from nannyml.base import Abstract1DResult
from nannyml.exceptions import InvalidArgumentsException
from nannyml.plots.blueprints.comparisons import ResultCompareMixin
from nannyml.plots.blueprints.metrics import plot_metric
from nannyml.usage_logging import UsageEvent, log_usage

Metric = namedtuple("Metric", "display_name column_name")


class Result(Abstract1DResult, ResultCompareMixin):
    """Contains the results of the data reconstruction drift calculation and provides plotting functionality."""

    def __init__(
        self,
        results_data: pd.DataFrame,
        column_names: List[str],
        categorical_column_names: List[str],
        continuous_column_names: List[str],
        timestamp_column_name: Optional[str] = None,
    ):
        super().__init__(results_data)

        self.column_names = column_names
        self.categorical_column_names = categorical_column_names
        self.continuous_column_names = continuous_column_names
        self.timestamp_column_name = timestamp_column_name
        self.metrics = [Metric(display_name='Reconstruction error', column_name='reconstruction_error')]

    def keys(self) -> List[Key]:
        return [Key(properties=('reconstruction_error',), display_names=('Reconstruction error',))]

    def _filter(self, period: str, metrics: Optional[List[str]] = None, *args, **kwargs) -> Result:
        if metrics is None:
            metrics = [metric.column_name for metric in self.metrics]

        data = pd.concat([self.data.loc[:, (['chunk'])], self.data.loc[:, (metrics,)]], axis=1)

        if period != 'all':
            data = data.loc[data[('chunk', 'period')] == period, :]

        data = data.reset_index(drop=True)
        result = copy.deepcopy(self)
        result.data = data

        return result

    @log_usage(UsageEvent.MULTIVAR_DRIFT_PLOT, metadata_from_kwargs=['kind'])
    def plot(self, kind: str = 'drift', *args, **kwargs) -> Optional[go.Figure]:
        """Renders plots for metrics returned by the multivariate data reconstruction calculator.

        The different plot kinds that are available:

        - ``drift``
                plots the multivariate reconstruction error over the provided features
                per :class:`~nannyml.chunk.Chunk`.

        Parameters
        ----------
        kind: str, default=`drift`
            The kind of plot you want to have. Value can currently only be ``drift``.
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
        >>> reference_df, analysis_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>>
        >>> column_names = [col for col in reference_df.columns
        >>>                         if col not in ['y_pred', 'y_pred_proba', 'work_home_actual', 'timestamp']]
        >>> calc = nml.DataReconstructionDriftCalculator(
        >>>     column_names=column_names,
        >>>     timestamp_column_name='timestamp'
        >>> )
        >>> calc.fit(reference_df)
        >>> results = calc.calculate(analysis_df)
        >>> print(results.data)  # access the numbers
                             key  start_index  ...  upper_threshold alert
        0       [0:4999]            0  ...         1.511762  True
        1    [5000:9999]         5000  ...         1.511762  True
        2  [10000:14999]        10000  ...         1.511762  True
        3  [15000:19999]        15000  ...         1.511762  True
        4  [20000:24999]        20000  ...         1.511762  True
        5  [25000:29999]        25000  ...         1.511762  True
        6  [30000:34999]        30000  ...         1.511762  True
        7  [35000:39999]        35000  ...         1.511762  True
        8  [40000:44999]        40000  ...         1.511762  True
        9  [45000:49999]        45000  ...         1.511762  True
        >>> fig = results.plot(plot_reference=True)
        >>> fig.show()
        """
        if kind == 'drift':
            return plot_metric(
                self,
                title='Multivariate drift (PCA reconstruction error)',
                metric_display_name='Data reconstruction drift',
                metric_column_name='reconstruction_error',
            )
        else:
            raise InvalidArgumentsException(f"unknown plot kind '{kind}'. " f"Please provide one of: ['drift'].")
