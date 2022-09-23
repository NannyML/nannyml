#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Contains the results of the univariate statistical drift calculation and provides plotting functionality."""
from __future__ import annotations

import copy
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

from nannyml._typing import Metric
from nannyml.base import AbstractCalculatorResult
from nannyml.chunk import Chunk
from nannyml.exceptions import InvalidArgumentsException
from nannyml.plots._joy_plot import _joy_plot
from nannyml.plots._stacked_bar_plot import _stacked_bar_plot
from nannyml.plots._step_plot import _step_plot


class Result(AbstractCalculatorResult):
    """Contains the results of the univariate statistical drift calculation and provides plotting functionality."""

    metric_to_col_suffix = {'KS': '_dstat', 'Chi2': '_chi2'}
    col_suffix_to_metric = {v: k for k, v in metric_to_col_suffix.items()}

    def __init__(self, results_data: pd.DataFrame, calculator):
        super().__init__(results_data)

        from nannyml.drift.model_inputs.univariate.statistical.calculator import UnivariateStatisticalDriftCalculator

        if not isinstance(calculator, UnivariateStatisticalDriftCalculator):
            raise RuntimeError(
                f"{calculator.__class__.__name__} is not an instance of type " f"UnivariateStatisticalDriftCalculator"
            )
        self.calculator = calculator

    def _filter(self, period: str, metrics: List[str] = None, *args, **kwargs) -> Result:
        columns = self.DEFAULT_COLUMNS

        if 'features' in kwargs:
            features = kwargs['features']
        else:
            features = self.calculator.feature_column_names

        if metrics is None:
            metrics = list(self.metric_to_col_suffix.keys())

        columns += [
            f'{feature}{self.metric_to_col_suffix["KS"]}'
            for feature in features
            if feature in self.calculator.continuous_column_names and 'KS' in metrics
        ]

        columns += [
            f'{feature}{self.metric_to_col_suffix["Chi2"]}'
            for feature in features
            if feature in self.calculator.categorical_column_names and 'Chi2' in metrics
        ]

        columns += [f'{feature}_alert' for feature in features]

        if period == 'all':
            data = self.data.loc[:, columns]
        else:
            data = self.data.loc[self.data['period'] == period, columns]

        return Result(results_data=data, calculator=copy.deepcopy(self.calculator))

    def _to_metric_list(self, period: str, metrics: List[str] = None, *args, **kwargs) -> List[Metric]:
        def _parse(column_name: str, calculator_name: str, start_date: datetime, end_date: datetime, value) -> Metric:
            idx = column_name.rindex('_')
            timestamp = start_date + (end_date - start_date) / 2

            return Metric(
                feature_name=column_name[0:idx],
                calculator_name=calculator_name,
                metric_name=self.col_suffix_to_metric[column_name[idx:]],
                timestamp=timestamp,
                value=value,
                upper_threshold=None,
                lower_threshold=None,
            )

        if self.calculator.timestamp_column_name is None:
            raise NotImplementedError(
                'no timestamp column was specified. Listing metrics currently requires a '
                'timestamp column to be specified and present'
            )

        res: List[Metric] = []

        if metrics is None:
            metrics = list(self.metric_to_col_suffix.keys())

        if 'features' in kwargs:
            features = kwargs['features']
        else:
            features = self.calculator.feature_column_names

        filtered = self.filter(period, metrics, features=features, *args, **kwargs).data

        for feature_metric_col in [
            col for col in filtered.columns if str(col).endswith(tuple(self.col_suffix_to_metric))
        ]:
            res += (
                filtered[['start_date', 'end_date', feature_metric_col]]
                .apply(lambda r: _parse(feature_metric_col, 'univariate statistical feature drift', *r), axis=1)
                .to_list()
            )

        return res

    def plot(
        self,
        kind: str = 'feature',
        metric: str = 'statistic',
        feature_column_name: str = None,
        plot_reference: bool = False,
        *args,
        **kwargs,
    ) -> Optional[go.Figure]:
        """Renders plots for metrics returned by the univariate statistical drift calculator.

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
        metric : str, default=``statistic``
            The metric to plot. Allowed values are ``statistic`` and ``p_value``.
            Not applicable when plotting distributions.
        feature_column_name : str
            Column name identifying a feature according to the preset model metadata. The function will raise an
            exception when no feature using that column name was found in the metadata.
            Either ``feature_column_name`` or ``feature_label`` should be specified.
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
        >>>
        >>> reference_df, analysis_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>>
        >>> feature_column_names = [col for col in reference_df.columns
        >>>                         if col not in ['y_pred', 'y_pred_proba', 'work_home_actual', 'timestamp']]
        >>> calc = nml.UnivariateStatisticalDriftCalculator(
        >>>     feature_column_names=feature_column_names,
        >>>     timestamp_column_name='timestamp'
        >>> )
        >>> calc.fit(reference_df)
        >>> results = calc.calculate(analysis_df)
        >>> print(results.data)  # check the numbers
                     key  start_index  ...  identifier_alert identifier_threshold
        0       [0:4999]            0  ...              True                 0.05
        1    [5000:9999]         5000  ...              True                 0.05
        2  [10000:14999]        10000  ...              True                 0.05
        3  [15000:19999]        15000  ...              True                 0.05
        4  [20000:24999]        20000  ...              True                 0.05
        5  [25000:29999]        25000  ...              True                 0.05
        6  [30000:34999]        30000  ...              True                 0.05
        7  [35000:39999]        35000  ...              True                 0.05
        8  [40000:44999]        40000  ...              True                 0.05
        9  [45000:49999]        45000  ...              True                 0.05
        >>> for feature in calc.feature_column_names:
        >>>     fig = results.plot(kind='feature_drift', metric='statistic', plot_reference=True,
        >>>                        feature_column_name=feature)
        >>>     fig.show()

        """
        if kind == 'feature_drift':
            if feature_column_name is None:
                raise InvalidArgumentsException(
                    "must specify a feature to plot " "using the 'feature_column_name' parameter"
                )
            return _feature_drift(self.data, self.calculator, feature_column_name, metric, plot_reference)
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
        if feature_column_name in self.calculator.continuous_column_names:
            return _plot_continuous_feature_distribution(
                analysis_data, drift_data, feature_column_name, self.calculator, plot_reference
            )
        if feature_column_name in self.calculator.categorical_column_names:
            return _plot_categorical_feature_distribution(
                analysis_data, drift_data, feature_column_name, self.calculator, plot_reference
            )


def _feature_drift(
    data: pd.DataFrame,
    calculator,
    feature: str,
    metric: str = 'statistic',
    plot_reference: bool = False,
) -> go.Figure:
    """Renders a line plot for a chosen metric of univariate statistical feature drift calculation results."""

    (
        metric_column_name,
        metric_label,
        threshold_column_name,
        drift_column_name,
        title,
    ) = _get_drift_column_names_for_feature(
        feature, metric, calculator.continuous_column_names, calculator.categorical_column_names
    )

    if not plot_reference:
        data = data.loc[data['period'] == 'analysis', :]

    is_time_based_x_axis = calculator.timestamp_column_name is not None

    fig = _step_plot(
        table=data,
        metric_column_name=metric_column_name,
        chunk_column_name='key',
        drift_column_name=drift_column_name,
        lower_threshold_column_name=threshold_column_name,
        hover_labels=['Chunk', metric_label, 'Target data'],
        title=title,
        y_axis_title=metric_label,
        v_line_separating_analysis_period=plot_reference,
        statistically_significant_column_name=drift_column_name,
        start_date_column_name='start_date' if is_time_based_x_axis else None,
        end_date_column_name='end_date' if is_time_based_x_axis else None,
    )
    return fig


def _get_drift_column_names_for_feature(
    feature_column_name: str,
    metric: str,
    continuous_feature_column_names: List[str],
    categorical_feature_column_names: List[str],
) -> Tuple:
    metric_column_name, metric_label, threshold_column_name = None, None, None
    if metric == 'statistic':
        if feature_column_name in categorical_feature_column_names:
            metric_column_name = f'{feature_column_name}_chi2'
            metric_label = 'Chi-square statistic'
        elif feature_column_name in continuous_feature_column_names:
            metric_column_name = f'{feature_column_name}_dstat'
            metric_label = 'KS statistic'
        threshold_column_name = None
    elif metric == 'p_value':
        metric_column_name = f'{feature_column_name}_p_value'
        metric_label = 'P-value'
        threshold_column_name = f'{feature_column_name}_threshold'

    drift_column_name = f'{feature_column_name}_alert'
    title = f'{metric_label} for {feature_column_name}'

    return metric_column_name, metric_label, threshold_column_name, drift_column_name, title


def _plot_continuous_feature_distribution(
    data: pd.DataFrame, drift_data: pd.DataFrame, feature_column_name: str, calculator, plot_reference: bool
) -> go.Figure:
    """Plots the data distribution and associated drift for each chunk of a given continuous feature."""
    from nannyml.drift.model_inputs.univariate.statistical.calculator import UnivariateStatisticalDriftCalculator

    if not isinstance(calculator, UnivariateStatisticalDriftCalculator):
        raise InvalidArgumentsException(
            f"{calculator.__class__.__name__} is not an instance of type " f"UnivariateStatisticalDriftCalculator"
        )

    if not plot_reference:
        drift_data = drift_data.loc[drift_data['period'] == 'analysis']

    x_axis_title = f'{feature_column_name}'
    drift_column_name = f'{feature_column_name}_alert'
    title = f'Distribution over time for {feature_column_name}'

    data['period'] = 'analysis'
    feature_table = _create_feature_table(calculator.chunker.split(data))

    if plot_reference:
        reference_feature_table = _create_feature_table(calculator.chunker.split(calculator.previous_reference_data))
        reference_feature_table['period'] = 'reference'
        feature_table = pd.concat([reference_feature_table, feature_table], ignore_index=True)

    is_time_based_x_axis = calculator.timestamp_column_name is not None

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
    data: pd.DataFrame, drift_data: pd.DataFrame, feature_column_name: str, calculator, plot_reference: bool
) -> go.Figure:
    """Plots the data distribution and associated drift for each chunk of a given categorical feature."""
    from nannyml.drift.model_inputs.univariate.statistical.calculator import UnivariateStatisticalDriftCalculator

    if not isinstance(calculator, UnivariateStatisticalDriftCalculator):
        raise InvalidArgumentsException(
            f"{calculator.__class__.__name__} is not an instance of type " f"UnivariateStatisticalDriftCalculator"
        )

    if not plot_reference:
        drift_data = drift_data.loc[drift_data['period'] == 'analysis']

    yaxis_title = f'{feature_column_name}'
    drift_column_name = f'{feature_column_name}_alert'
    title = f'Distribution over time for {feature_column_name}'

    data['period'] = 'analysis'
    feature_table = _create_feature_table(calculator.chunker.split(data))

    if plot_reference:
        reference_feature_table = _create_feature_table(calculator.chunker.split(calculator.previous_reference_data))
        reference_feature_table['period'] = 'reference'
        feature_table = pd.concat([reference_feature_table, feature_table], ignore_index=True)

    is_time_based_x_axis = calculator.timestamp_column_name is not None

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
