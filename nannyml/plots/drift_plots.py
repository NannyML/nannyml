#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  #
#  License: Apache Software License 2.0

#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Contains plots and utilities for plotting drift results."""
import pandas as pd
import plotly.graph_objects as go

from nannyml import FeatureType, InvalidArgumentsException, ModelMetadata
from nannyml.plots._line_plot import _line_plot

_CHUNK_KEY_COLUMN_NAME = 'key'
_THRESHOLD_COLUMN_NAME = 'threshold'


class DriftPlots:
    def __init__(self, model_metadata: ModelMetadata):
        self._metadata = model_metadata

    def plot_univariate_statistical_drift(
        self,
        drift_results: pd.DataFrame,
        metric: str = 'statistic',
        feature_label: str = None,
        feature_column_name: str = None,
    ) -> go.Figure:
        if feature_label is None and feature_column_name is None:
            raise InvalidArgumentsException("one of 'feature_label' or 'feature_column_name' should be provided.")

        feature = (
            self._metadata.feature(feature=feature_label)
            if feature_label
            else self._metadata.feature(column=feature_column_name)
        )

        if feature is None:
            raise InvalidArgumentsException(f'could not find a feature {feature_label or feature_column_name} ')
        metric_column_name = ''
        if metric == 'statistic':
            if feature.feature_type == FeatureType.CATEGORICAL:
                metric_column_name = f'{feature.column_name}_chi2'
            elif feature.feature_type == FeatureType.CONTINUOUS:
                metric_column_name = f'{feature.column_name}_dstat'
        elif metric == 'p_value':
            metric_column_name = f'{feature.column_name}_p_value'

        drift_column_name = f'{feature.column_name}_alert'
        title = f'{metric} value for {feature_label}'

        fig = _line_plot(
            table=drift_results,
            metric_column_name=metric_column_name,
            chunk_column_name=_CHUNK_KEY_COLUMN_NAME,
            drift_column_name=drift_column_name,
            threshold_column_name=_THRESHOLD_COLUMN_NAME,
            title=title,
        )

        return fig


def plot_univariate_statistical_drift(drift_results: pd.DataFrame, feature: str, metric: str):
    """Plots results of univariate statistical drift calculations."""
    if drift_results.attrs.get('nml_drift_calculator') != "nannyml.drift.univariate_statistical_drift_calculator":
        raise InvalidArgumentsException(
            'given drift results are not results of ' 'univariate statistical drift detection.'
        )

    metric_related_columns = [column_name for column_name in drift_results.columns if feature in column_name]
    if len(metric_related_columns) == 0:
        raise InvalidArgumentsException(f'could not find feature `{feature}` in drift results')

    if metric == 'statistic':
        metric_column_name = f'{feature}_dstat' if f'{feature}_dstat' in metric_related_columns else f'{feature}_chi2'
    elif metric == 'p_value':
        metric_column_name = f'{feature}_p_value'
        drift_results['threshold'] = 0.05

    fig = _line_plot(
        table=drift_results,
        metric_column_name=metric_column_name,
        chunk_column_name='key',
        drift_column_name=f'{feature}_alert',
        threshold_column_name='threshold',
    )
    fig.show()
