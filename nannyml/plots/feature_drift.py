#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from typing import List, Tuple

import pandas as pd
import plotly.graph_objects as go

from nannyml.drift.model_inputs.univariate.statistical import UnivariateStatisticalDriftCalculatorResult
from nannyml.plots._step_plot import _step_plot


def feature_drift(
    results: UnivariateStatisticalDriftCalculatorResult,
    feature: str,
    metric: str = 'statistic',
    reference_data: pd.DataFrame = None,
) -> go.Figure:
    """Renders a line plot for a chosen metric of univariate statistical feature drift calculation results."""

    (
        metric_column_name,
        metric_label,
        threshold_column_name,
        drift_column_name,
        title,
    ) = _get_drift_column_names_for_feature(
        feature_column_name=feature,
        metric=metric,
        continuous_feature_column_names=results.calculator.continuous_column_names,
        categorical_feature_column_names=results.calculator.categorical_column_names,
    )

    plot_period_separator = False

    if reference_data is not None:
        plot_period_separator = True
        reference_results = results.calculator.calculate(reference_data)
        reference_results.data['period'] = 'reference'
        data = pd.concat([reference_results.data, results.data], ignore_index=True)
    else:
        data = results.data

    fig = _step_plot(
        table=data,
        metric_column_name=metric_column_name,
        chunk_column_name='key',
        drift_column_name=drift_column_name,
        lower_threshold_column_name=threshold_column_name,
        hover_labels=['Chunk', metric_label, 'Target data'],
        title=title,
        y_axis_title=metric_label,
        v_line_separating_analysis_period=plot_period_separator,
        statistically_significant_column_name=drift_column_name,
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
