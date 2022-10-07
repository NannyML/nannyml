#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Contains the results of the univariate statistical drift calculation and provides plotting functionality."""
from typing import List, Optional, Union

import pandas as pd
import plotly.graph_objects as go

from nannyml.base import AbstractCalculatorResult, _column_is_continuous
from nannyml.chunk import Chunk
from nannyml.drift.model_inputs.univariate.distance.metrics import Metric, MetricFactory
from nannyml.exceptions import InvalidArgumentsException
from nannyml.plots._joy_plot import _joy_plot
from nannyml.plots._stacked_bar_plot import _stacked_bar_plot
from nannyml.plots._step_plot import _step_plot


class Result(AbstractCalculatorResult):
    """Contains the results of the univariate statistical drift calculation and provides plotting functionality."""

    def __init__(self, results_data: pd.DataFrame, calculator):
        super().__init__(results_data)

        from nannyml.drift.model_inputs.univariate.distance.calculator import DistanceDriftCalculator

        if not isinstance(calculator, DistanceDriftCalculator):
            raise RuntimeError(
                f"{calculator.__class__.__name__} is not an instance of type " f"UnivariateStatisticalDriftCalculator"
            )
        self.calculator = calculator

    def _filter(self, period: str, metrics: List[str] = None, *args, **kwargs) -> AbstractCalculatorResult:
        pass

    def plot(
        self,
        kind: str = 'feature_drift',
        metric: Union[str, Metric] = None,
        feature_column_name: str = None,
        plot_reference: bool = False,
        *args,
        **kwargs,
    ) -> Optional[go.Figure]:
        """Renders plots for metrics returned by the univariate distance drift calculator.

        For any feature you can render the statistic value or p-values as a step plot, or create a distribution plot.
        Select a plot using the ``kind`` parameter:

        - ``feature_drift``
                plots drift per :class:`~nannyml.chunk.Chunk` for a single feature of a chunked data set.
        - ``feature_distribution``
                plots feature distribution per :class:`~nannyml.chunk.Chunk`.
                Joyplot for continuous features, stacked bar charts for categorical features.

        Parameters
        ----------
        kind: str, default=`feature_drift`
            The kind of plot you want to have. Allowed values are `feature_drift`` and ``feature_distribution``.
        feature_column_name : str
            Column name identifying a feature according to the preset model metadata. The function will raise an
            exception when no feature using that column name was found in the metadata.
            Either ``feature_column_name`` or ``feature_label`` should be specified.
        metric: str
            The name of the metric to plot. Allowed values are [`jensen_shannon`].
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
        # TODO: add example

        """
        if metric is None:
            raise InvalidArgumentsException(
                "no value for 'metric' given. Please provide the name of a metric to display."
            )
        if kind == 'feature_drift':
            if feature_column_name is None:
                raise InvalidArgumentsException(
                    "must specify a feature to plot " "using the 'feature_column_name' parameter"
                )
            return self.plot_feature_drift(metric, feature_column_name, plot_reference)
        elif kind == 'feature_distribution':
            if feature_column_name is None:
                raise InvalidArgumentsException(
                    "must specify a feature to plot " "using the 'feature_column_name' parameter"
                )
            return self._plot_feature_distribution(
                analysis_data=self.calculator.previous_analysis_data,
                plot_reference=plot_reference,
                drift_data=self.data,
                feature_column_name=feature_column_name,
                metric=metric,
            )
        else:
            raise InvalidArgumentsException(
                f"unknown plot kind '{kind}'. " f"Please provide on of: ['feature_drift', 'feature_distribution']."
            )

    def _plot_feature_distribution(
        self,
        analysis_data: pd.DataFrame,
        drift_data: pd.DataFrame,
        feature_column_name: str,
        metric: Union[str, Metric],
        plot_reference: bool,
    ) -> go.Figure:
        """Plots the data distribution and associated drift for each chunk of a given continuous feature."""
        if _column_is_continuous(analysis_data[feature_column_name]):
            return self._plot_continuous_feature_distribution(
                analysis_data, drift_data, feature_column_name, metric, plot_reference
            )
        else:
            return self._plot_categorical_feature_distribution(
                analysis_data, drift_data, feature_column_name, metric, plot_reference
            )
        # pass

    def plot_feature_drift(
        self,
        metric: Union[str, Metric],
        feature: str,
        plot_reference: bool = False,
    ) -> go.Figure:
        """Renders a line plot for a chosen metric of univariate statistical feature drift calculation results."""
        result_data = self.data.copy()

        if isinstance(metric, str):
            metric = MetricFactory.create(key=metric, kwargs={'calculator': self.calculator})

        if not plot_reference:
            result_data = result_data[result_data['period'] == 'analysis']

        is_time_based_x_axis = self.calculator.timestamp_column_name is not None

        fig = _step_plot(
            table=result_data,
            metric_column_name=f'{feature}_{metric.column_name}',
            chunk_column_name='key',
            start_date_column_name='start_date' if is_time_based_x_axis else None,
            end_date_column_name='end_date' if is_time_based_x_axis else None,
            drift_column_name=f'{feature}_{metric.column_name}_alert',
            lower_threshold_column_name=f'{feature}_{metric.column_name}_lower_threshold',
            upper_threshold_column_name=f'{feature}_{metric.column_name}_upper_threshold',
            hover_labels=['Chunk', f'{metric.display_name}', 'Target data'],
            title=f'{metric.display_name} distance for {feature}',
            y_axis_title=f'{metric.display_name}',
            v_line_separating_analysis_period=plot_reference,
        )
        return fig

    def _plot_continuous_feature_distribution(
        self,
        data: pd.DataFrame,
        drift_data: pd.DataFrame,
        feature_column_name: str,
        metric: Union[str, Metric],
        plot_reference: bool,
    ) -> go.Figure:
        """Plots the data distribution and associated drift for each chunk of a given continuous feature."""
        if isinstance(metric, str):
            metric = MetricFactory.create(key=metric, kwargs={'calculator': self.calculator})

        if not plot_reference:
            drift_data = drift_data.loc[drift_data['period'] == 'analysis']

        x_axis_title = f'{feature_column_name}'
        drift_column_name = f'{feature_column_name}_{metric.column_name}_alert'
        title = f'Distribution over time for {feature_column_name}'

        data['period'] = 'analysis'
        feature_table = _create_feature_table(self.calculator.chunker.split(data))

        if plot_reference:
            reference_feature_table = _create_feature_table(
                self.calculator.chunker.split(self.calculator.previous_reference_data)
            )
            reference_feature_table['period'] = 'reference'
            feature_table = pd.concat([reference_feature_table, feature_table], ignore_index=True)

        is_time_based_x_axis = self.calculator.timestamp_column_name is not None

        fig = _joy_plot(
            feature_table=feature_table,
            drift_table=drift_data,
            chunk_column_name='key',
            drift_column_name=drift_column_name,
            feature_column_name=feature_column_name,
            x_axis_title=x_axis_title,
            title=title,
            style='vertical',
            start_date_column_name='start_date' if is_time_based_x_axis else None,
            end_date_column_name='end_date' if is_time_based_x_axis else None,
        )
        return fig

    def _plot_categorical_feature_distribution(
        self,
        data: pd.DataFrame,
        drift_data: pd.DataFrame,
        feature_column_name: str,
        metric: Union[str, Metric],
        plot_reference: bool,
    ) -> go.Figure:
        """Plots the data distribution and associated drift for each chunk of a given categorical feature."""
        if isinstance(metric, str):
            metric = MetricFactory.create(key=metric, kwargs={'calculator': self.calculator})

        if not plot_reference:
            drift_data = drift_data.loc[drift_data['period'] == 'analysis']

        yaxis_title = f'{feature_column_name}'
        drift_column_name = f'{feature_column_name}_alert'
        title = f'Distribution over time for {feature_column_name}'

        data['period'] = 'analysis'
        feature_table = _create_feature_table(self.calculator.chunker.split(data))

        if plot_reference:
            reference_feature_table = _create_feature_table(
                self.calculator.chunker.split(self.calculator.previous_reference_data)
            )
            reference_feature_table['period'] = 'reference'
            feature_table = pd.concat([reference_feature_table, feature_table], ignore_index=True)

        is_time_based_x_axis = self.calculator.timestamp_column_name is not None

        fig = _stacked_bar_plot(
            feature_table=feature_table,
            drift_table=drift_data,
            chunk_column_name='key',
            drift_column_name=drift_column_name,
            feature_column_name=feature_column_name,
            yaxis_title=yaxis_title,
            title=title,
            start_date_column_name='start_date' if is_time_based_x_axis else None,
            end_date_column_name='end_date' if is_time_based_x_axis else None,
        )
        return fig


def _create_feature_table(chunks: List[Chunk]) -> pd.DataFrame:
    return pd.concat([chunk.data.assign(key=chunk.key) for chunk in chunks])
