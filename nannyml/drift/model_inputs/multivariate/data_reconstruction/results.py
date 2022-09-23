#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Contains the results of the data reconstruction drift calculation and provides plotting functionality."""
from __future__ import annotations

import copy
from datetime import datetime
from typing import Optional, List

import pandas as pd
import plotly.graph_objects as go

from nannyml._typing import Metric
from nannyml.base import AbstractCalculator, AbstractCalculatorResult
from nannyml.exceptions import InvalidArgumentsException
from nannyml.plots._step_plot import _step_plot


class Result(AbstractCalculatorResult):
    """Contains the results of the data reconstruction drift calculation and provides plotting functionality."""

    metric_label_to_col = {'reconstruction error': 'reconstruction_error'}
    col_to_metric_label = {v: k for k, v in metric_label_to_col.items()}

    def __init__(self, results_data: pd.DataFrame, calculator: AbstractCalculator):
        super().__init__(results_data)

        from . import DataReconstructionDriftCalculator

        if not isinstance(calculator, DataReconstructionDriftCalculator):
            raise RuntimeError(
                f"{calculator.__class__.__name__} is not an instance of type " f"DataReconstructionDriftCalculator"
            )
        self.calculator = calculator

    def _filter(self, period: str, metrics: List[str] = None, *args, **kwargs) -> Result:
        columns = self.DEFAULT_COLUMNS

        if metrics is None:
            metrics = list(self.col_to_metric_label.keys())

        columns += [metric for metric in metrics]

        columns += ['alert']

        if period == 'all':
            data = self.data.loc[:, columns]
        else:
            data = self.data.loc[self.data['period'] == period, columns]

        return Result(results_data=data, calculator=copy.deepcopy(self.calculator))

    def _to_metric_list(self, period: str, metrics: List[str] = None, *args, **kwargs) -> List[Metric]:
        def _parse(column_name: str, start_date: datetime, end_date: datetime, value) -> Metric:
            timestamp = start_date + (end_date - start_date) / 2

            return Metric(
                feature_name=None,
                metric_name=self.col_to_metric_label[column_name],
                timestamp=timestamp,
                value=value,
            )

        if self.calculator.timestamp_column_name is None:
            raise NotImplementedError(
                'no timestamp column was specified. Listing metrics currently requires a '
                'timestamp column to be specified and present'
            )

        res: List[Metric] = []

        if metrics is None:
            metrics = list(self.col_to_metric_label.keys())

        filtered = self.filter(period, metrics, *args, **kwargs).data

        for metric_col in metrics:
            res += (
                filtered[['start_date', 'end_date', metric_col]]
                .apply(lambda r: _parse(metric_col, *r), axis=1)
                .to_list()
            )

        return res

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
        >>> feature_column_names = [col for col in reference_df.columns
        >>>                         if col not in ['y_pred', 'y_pred_proba', 'work_home_actual', 'timestamp']]
        >>> calc = nml.DataReconstructionDriftCalculator(
        >>>     feature_column_names=feature_column_names,
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
            return _plot_drift(self.data, self.calculator, plot_reference)
        else:
            raise InvalidArgumentsException(f"unknown plot kind '{kind}'. " f"Please provide one of: ['drift'].")

    # @property
    # def plots(self) -> Dict[str, go.Figure]:
    #     return {'multivariate_feature_drift': _plot_drift(self.data)}


def _plot_drift(data: pd.DataFrame, calculator, plot_reference: bool) -> go.Figure:
    plot_period_separator = plot_reference

    data['period'] = 'analysis'

    if plot_reference:
        reference_results = calculator.previous_reference_results.copy()
        reference_results['period'] = 'reference'
        data = pd.concat([reference_results, data], ignore_index=True)

    is_time_based_x_axis = calculator.timestamp_column_name is not None

    fig = _step_plot(
        table=data,
        metric_column_name='reconstruction_error',
        chunk_column_name='key',
        drift_column_name='alert',
        lower_threshold_column_name='lower_threshold',
        upper_threshold_column_name='upper_threshold',
        hover_labels=['Chunk', 'Reconstruction error', 'Target data'],
        title='Data Reconstruction Drift',
        y_axis_title='Reconstruction Error',
        v_line_separating_analysis_period=plot_period_separator,
        sampling_error_column_name='sampling_error',
        lower_confidence_column_name='lower_confidence_bound',
        upper_confidence_column_name='upper_confidence_bound',
        plot_confidence_for_reference=True,
        start_date_column_name='start_date' if is_time_based_x_axis else None,
        end_date_column_name='end_date' if is_time_based_x_axis else None,
    )

    return fig
