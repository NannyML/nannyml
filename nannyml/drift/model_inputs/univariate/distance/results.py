#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Contains the results of the univariate statistical drift calculation and provides plotting functionality."""
from typing import List

import pandas as pd
import plotly.graph_objects as go

from nannyml.base import AbstractCalculatorResult
from nannyml.chunk import Chunk
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
        metric: str = 'jensen_shannon',
        kind: str = 'feature_drift',
        feature_column_name: str = None,
        plot_reference: bool = False,
        *args,
        **kwargs,
    ) -> go.Figure:
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
            The name of the metric to plot. Allowed values are 'jensen_shannon'.
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
        plot_reference: bool,
    ) -> go.Figure:
        """Plots the data distribution and associated drift for each chunk of a given continuous feature."""
        # if feature_column_name in self.calculator.continuous_column_names:
        #     return _plot_continuous_feature_distribution(
        #         analysis_data, drift_data, feature_column_name, self.calculator, plot_reference
        #     )
        # if feature_column_name in self.calculator.categorical_column_names:
        #     return _plot_categorical_feature_distribution(
        #         analysis_data, drift_data, feature_column_name, self.calculator, plot_reference
        #     )
        pass

    def plot_feature_drift(
        self,
        metric: str,
        feature: str,
        plot_reference: bool = False,
    ) -> go.Figure:
        """Renders a line plot for a chosen metric of univariate statistical feature drift calculation results."""
        result_data = self.data.copy()

        if not plot_reference:
            result_data = result_data[result_data['period'] == 'analysis']

        is_time_based_x_axis = self.calculator.timestamp_column_name is not None

        fig = _step_plot(
            table=result_data,
            metric_column_name=f'{feature}_{metric}',
            chunk_column_name='key',
            start_date_column_name='start_date' if is_time_based_x_axis else None,
            end_date_column_name='end_date' if is_time_based_x_axis else None,
            drift_column_name=f'{feature}_alert',
            lower_threshold_column_name=f'{feature}_threshold',
            hover_labels=['Chunk', f'{metric}', 'Target data'],
            title=f'{metric} distance for {feature}',
            y_axis_title=f'{metric}',
            v_line_separating_analysis_period=plot_reference,
        )
        return fig


def _plot_continuous_feature_distribution(
    data: pd.DataFrame, drift_data: pd.DataFrame, feature_column_name: str, calculator, plot_reference: bool
) -> go.Figure:
    """Plots the data distribution and associated drift for each chunk of a given continuous feature."""
    from nannyml.drift.model_inputs.univariate.statistical.calculator import UnivariateStatisticalDriftCalculator

    if not isinstance(calculator, UnivariateStatisticalDriftCalculator):
        raise InvalidArgumentsException(
            f"{calculator.__class__.__name__} is not an instance of type " f"UnivariateStatisticalDriftCalculator"
        )

    x_axis_title = f'{feature_column_name}'
    drift_column_name = f'{feature_column_name}_alert'
    title = f'Distribution over time for {feature_column_name}'

    drift_data['period'] = 'analysis'
    data['period'] = 'analysis'

    feature_table = _create_feature_table(calculator.chunker.split(data, calculator.timestamp_column_name))

    if plot_reference:
        reference_drift = calculator.previous_reference_results
        if reference_drift is None:
            raise RuntimeError(
                f"could not plot continuous distribution for feature '{feature_column_name}': "
                f"calculator is missing reference results\n{calculator}"
            )
        reference_drift['period'] = 'reference'
        drift_data = pd.concat([reference_drift, drift_data], ignore_index=True)

        reference_feature_table = _create_feature_table(
            calculator.chunker.split(calculator.previous_reference_data, calculator.timestamp_column_name)
        )
        feature_table = pd.concat([reference_feature_table, feature_table], ignore_index=True)

    fig = _joy_plot(
        feature_table=feature_table,
        drift_table=drift_data,
        chunk_column_name='key',
        drift_column_name=drift_column_name,
        feature_column_name=feature_column_name,
        x_axis_title=x_axis_title,
        title=title,
        style='vertical',
    )
    return fig


def _plot_categorical_feature_distribution(
    data: pd.DataFrame, drift_data: pd.DataFrame, feature_column_name: str, calculator, plot_reference: bool
) -> go.Figure:
    """Plots the data distribution and associated drift for each chunk of a given categorical feature."""
    from nannyml.drift.model_inputs.univariate.statistical.calculator import UnivariateStatisticalDriftCalculator

    if not isinstance(calculator, UnivariateStatisticalDriftCalculator):
        raise InvalidArgumentsException(
            f"{calculator.__class__.__name__} is not an instance of type " f"UnivariateStatisticalDriftCalculator"
        )

    yaxis_title = f'{feature_column_name}'
    drift_column_name = f'{feature_column_name}_alert'
    title = f'Distribution over time for {feature_column_name}'

    drift_data['period'] = 'analysis'
    data['period'] = 'analysis'

    feature_table = _create_feature_table(calculator.chunker.split(data, calculator.timestamp_column_name))

    if plot_reference:
        reference_drift = calculator.previous_reference_results
        if reference_drift is None:
            raise RuntimeError(
                f"could not plot categorical distribution for feature '{feature_column_name}': "
                f"calculator is missing reference results\n{calculator}"
            )
        reference_drift['period'] = 'reference'
        drift_data = pd.concat([reference_drift, drift_data], ignore_index=True)

        reference_feature_table = _create_feature_table(
            calculator.chunker.split(calculator.previous_reference_data, calculator.timestamp_column_name)
        )
        feature_table = pd.concat([reference_feature_table, feature_table], ignore_index=True)

    fig = _stacked_bar_plot(
        feature_table=feature_table,
        drift_table=drift_data,
        chunk_column_name='key',
        drift_column_name=drift_column_name,
        feature_column_name=feature_column_name,
        yaxis_title=yaxis_title,
        title=title,
    )
    return fig


def _create_feature_table(chunks: List[Chunk]) -> pd.DataFrame:
    return pd.concat([chunk.data.assign(key=chunk.key) for chunk in chunks])
