#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Contains the results of the data reconstruction drift calculation and provides plotting functionality."""
from __future__ import annotations

import copy
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go

from nannyml.base import AbstractCalculatorResult
from nannyml.exceptions import InvalidArgumentsException
from nannyml.plots._step_plot import _step_plot


class Result(AbstractCalculatorResult):
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
        self.metrics = ['reconstruction_error']

    def _filter(self, period: str, metrics: List[str] = None, *args, **kwargs) -> Result:
        if metrics is None:
            metrics = self.metrics

        data = pd.concat([self.data.loc[:, (['chunk'])], self.data.loc[:, (metrics,)]], axis=1)

        if period == 'all':
            data = data.loc[:, :]
        else:
            data = data.loc[self.data.loc[:, ('chunk', 'period')] == period, :]

        data = data.reset_index(drop=True)
        result = copy.deepcopy(self)
        result.data = data

        return result

    def plot(self, kind: str = 'drift', plot_reference: bool = False, *args, **kwargs) -> Optional[go.Figure]:
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
            return self._plot_drift(self.to_df(multilevel=False), plot_reference)
        else:
            raise InvalidArgumentsException(f"unknown plot kind '{kind}'. " f"Please provide one of: ['drift'].")

    def _plot_drift(self, data: pd.DataFrame, plot_reference: bool) -> go.Figure:
        plot_period_separator = plot_reference

        if not plot_reference:
            data = data.loc[data['chunk_period'] == 'analysis', :]

        is_time_based_x_axis = self.timestamp_column_name is not None

        fig = _step_plot(
            table=data,
            metric_column_name='reconstruction_error_value',
            chunk_column_name='chunk_key',
            chunk_type_column_name='chunk_period',
            chunk_index_column_name='chunk_index',
            drift_column_name='reconstruction_error_alert',
            sampling_error_column_name='reconstruction_error_sampling_error',
            lower_threshold_column_name='reconstruction_error_lower_threshold',
            upper_threshold_column_name='reconstruction_error_upper_threshold',
            hover_labels=['Chunk', 'Reconstruction error', 'Target data'],
            title='Data Reconstruction Drift',
            y_axis_title='Reconstruction Error',
            v_line_separating_analysis_period=plot_period_separator,
            lower_confidence_column_name='reconstruction_error_lower_confidence_boundary',
            upper_confidence_column_name='reconstruction_error_upper_confidence_boundary',
            plot_confidence_for_reference=True,
            start_date_column_name='chunk_start_date' if is_time_based_x_axis else None,
            end_date_column_name='chunk_end_date' if is_time_based_x_axis else None,
        )

        return fig
