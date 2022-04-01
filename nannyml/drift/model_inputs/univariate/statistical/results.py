#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing univariate statistical drift calculation results and associated plotting implementations."""

from typing import List, Tuple

import pandas as pd
import plotly.graph_objects as go

from nannyml.chunk import Chunk
from nannyml.drift.base import DriftResult
from nannyml.exceptions import InvalidArgumentsException
from nannyml.metadata import Feature, FeatureType, ModelMetadata
from nannyml.plots import CHUNK_KEY_COLUMN_NAME
from nannyml.plots._joy_plot import _joy_plot
from nannyml.plots._stacked_bar_plot import _stacked_bar_plot
from nannyml.plots._step_plot import _step_plot


class UnivariateDriftResult(DriftResult):
    """Contains the univariate statistical drift calculation results and provides additional plotting functionality."""

    # TODO: this is messing up functionality in scratch files (sets runtime class to DataFrame). Check this!
    def __repr__(self):
        """Represent the DriftResults object as the data it contains."""
        return self.data.__repr__()

    def plot(
        self,
        kind: str = 'feature',
        metric: str = 'statistic',
        feature_label: str = None,
        feature_column_name: str = None,
        *args,
        **kwargs,
    ) -> go.Figure:
        """Renders a line plot for a chosen metric of statistical statistical drift calculation results.

        Given either a feature label (check ``model_metadata.features``) or the actual feature column name
        and a metric (one of either ``statistic`` or ``p_value``) this function will render a line plot displaying
        the metric value for the selected feature per chunk.
        Chunks are set on a time-based X-axis by using the period containing their observations.
        Chunks of different partitions (``reference`` and ``analysis``) are represented using different colors and
        a vertical separation if the drift results contain multiple partitions.

        The different plot kinds that are available:

        - ``feature_drift``: plots drift per :class:`~nannyml.chunk.Chunk` for a single feature of a chunked data set.
        - ``prediction_drift``: plots drift per :class:`~nannyml.chunk.Chunk` for the predictions of a chunked data set.
        - ``feature_distribution``: plots feature distribution per :class:`~nannyml.chunk.Chunk`.
          Joyplot for continuous features, stacked bar charts for categorical features.
        - ``prediction_distribution``: plots the prediction distribution per :class:`~nannyml.chunk.Chunk` of a chunked
          data set as a joyplot.


        Parameters
        ----------
        kind: str, default=`feature_drift`
            The kind of plot you want to have. Value must be one of ``feature_drift``, ``prediction_drift``,
            ``feature_distribution`` or ``prediction_distribution``.
        metric : str, default=``statistic``
            The metric to plot. Value must be one of ``statistic`` or ``p_value``
        feature_label : str
            Feature label identifying a feature according to the preset model metadata. The function will raise an
            exception when no feature of that label was found in the metadata.
            Either ``feature_label`` or ``feature_column_name`` should be specified.
        feature_column_name : str
            Column name identifying a feature according to the preset model metadata. The function will raise an
            exception when no feature using that column name was found in the metadata.
            Either ``feature_column_name`` or ``feature_label`` should be specified.

        Returns
        -------
        fig: plotly.graph_objects.Figure
            A ``Figure`` object containing the requested drift plot. Can be saved to disk or shown rendered on screen
            using ``fig.show()``.

        """
        if kind == 'feature_drift':
            feature = _get_feature(self.metadata, feature_label, feature_column_name)
            return _plot_feature_drift(self.data, feature, metric, args, kwargs)
        elif kind == 'prediction_drift':
            return _plot_prediction_drift(self.data, self.metadata.predicted_probability_column_name, metric)
        elif kind == 'feature_distribution':
            feature = _get_feature(self.metadata, feature_label, feature_column_name)
            return _plot_feature_distribution(
                data=self._analysis_data,
                drift_data=self.data,
                feature=feature,
            )
        elif kind == 'prediction_distribution':
            return _plot_prediction_distribution(
                data=self._analysis_data,
                drift_data=self.data,
                metadata=self.metadata,
            )
        else:
            raise InvalidArgumentsException(
                f"unknown plot kind '{kind}'. "
                f"Please provide on of: ['feature_drift', 'feature_distribution', "
                f"'prediction_drift', 'prediction_distribution']."
            )


def _get_feature(model_metadata: ModelMetadata, feature_label: str = None, feature_column_name: str = None) -> Feature:
    if feature_label is None and feature_column_name is None:
        raise InvalidArgumentsException("one of 'feature_label' or 'feature_column_name' should be provided.")

    feature = (
        model_metadata.feature(feature=feature_label)
        if feature_label
        else model_metadata.feature(column=feature_column_name)
    )

    if feature is None:
        raise InvalidArgumentsException(f'could not find a feature {feature_label or feature_column_name}')

    return feature


def _get_drift_column_names_for_feature(feature_column_name: str, feature_type: FeatureType, metric: str) -> Tuple:
    metric_column_name, metric_label, threshold_column_name = None, None, None
    if metric == 'statistic':
        if feature_type == FeatureType.CATEGORICAL:
            metric_column_name = f'{feature_column_name}_chi2'
            metric_label = 'Chi-square statistic'
        elif feature_type == FeatureType.CONTINUOUS:
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


def _plot_feature_drift(data: pd.DataFrame, feature: Feature, metric: str = 'statistic', *args, **kwargs) -> go.Figure:
    """Renders a line plot for a chosen metric of statistical statistical drift calculation results."""
    (
        metric_column_name,
        metric_label,
        threshold_column_name,
        drift_column_name,
        title,
    ) = _get_drift_column_names_for_feature(feature.column_name, feature.feature_type, metric)

    plot_partition_separator = len(data.value_counts()) > 1

    fig = _step_plot(
        table=data,
        metric_column_name=metric_column_name,
        chunk_column_name=CHUNK_KEY_COLUMN_NAME,
        drift_column_name=drift_column_name,
        threshold_column_name=threshold_column_name,
        hover_labels=['Chunk', metric_label, 'Target data'],
        title=title,
        y_axis_title=metric_label,
        v_line_separating_analysis_period=plot_partition_separator,
        statistically_significant_column_name=drift_column_name,
    )
    return fig


def _plot_prediction_drift(
    data: pd.DataFrame,
    prediction_column_name: str,
    metric: str = 'statistic',
) -> go.Figure:
    """Renders a line plot of the drift metric for a given feature."""
    (
        metric_column_name,
        metric_label,
        threshold_column_name,
        drift_column_name,
        title,
    ) = _get_drift_column_names_for_feature(prediction_column_name, FeatureType.CONTINUOUS, metric)

    plot_partition_separator = len(data.value_counts()) > 1

    fig = _step_plot(
        table=data,
        metric_column_name=metric_column_name,
        chunk_column_name=CHUNK_KEY_COLUMN_NAME,
        drift_column_name=drift_column_name,
        threshold_column_name=threshold_column_name,
        hover_labels=['Chunk', metric_label, 'Target data'],
        title=title,
        y_axis_title=metric_label,
        v_line_separating_analysis_period=plot_partition_separator,
        statistically_significant_column_name=drift_column_name,
    )
    return fig


def _plot_feature_distribution(data: List[Chunk], drift_data: pd.DataFrame, feature: Feature) -> go.Figure:
    """Plots the data distribution and associated drift for each chunk of a given continuous feature."""
    if feature.feature_type is FeatureType.CONTINUOUS:
        return _plot_continuous_feature_distribution(data, drift_data, feature)
    elif feature.feature_type is FeatureType.CATEGORICAL:
        return _plot_categorical_feature_distribution(data, drift_data, feature)


def _plot_continuous_feature_distribution(data: List[Chunk], drift_data: pd.DataFrame, feature: Feature) -> go.Figure:
    """Plots the data distribution and associated drift for each chunk of a given continuous feature."""
    feature_column_name = feature.column_name
    x_axis_title = f'{feature_column_name}'
    drift_column_name = f'{feature_column_name}_alert'
    title = f'Distribution over time for {feature.label}'

    fig = _joy_plot(
        feature_table=_create_feature_table(data=data),
        drift_table=drift_data,
        chunk_column_name=CHUNK_KEY_COLUMN_NAME,
        drift_column_name=drift_column_name,
        feature_column_name=feature_column_name,
        x_axis_title=x_axis_title,
        title=title,
        style='vertical',
    )
    return fig


def _plot_categorical_feature_distribution(data: List[Chunk], drift_data: pd.DataFrame, feature: Feature) -> go.Figure:
    """Plots the data distribution and associated drift for each chunk of a given categorical feature."""
    feature_column_name = feature.column_name
    x_axis_title = f'{feature_column_name}'
    drift_column_name = f'{feature_column_name}_alert'
    title = f'Distribution over time for {feature.label}'

    fig = _stacked_bar_plot(
        feature_table=_create_feature_table(data=data),
        drift_table=drift_data,
        chunk_column_name=CHUNK_KEY_COLUMN_NAME,
        drift_column_name=drift_column_name,
        feature_column_name=feature_column_name,
        x_axis_title=x_axis_title,
        title=title,
    )
    return fig


def _plot_prediction_distribution(
    data: List[Chunk],
    drift_data: pd.DataFrame,
    metadata: ModelMetadata,
) -> go.Figure:
    """Plots the data distribution and associated drift for each chunk of the model predictions.

    Parameters
    ----------
    data : pd.DataFrame
        The original model inputs and outputs
    drift_data : pd.DataFrame
        The results of the drift calculation
    metadata: ModelMetadata
        The metadata for the monitored model

    Returns
    -------
    fig: plotly.graph_objects.Figure
        A visualization of the data distribution and drift using joy-plots.
    """
    predicted_probability_column_name = metadata.predicted_probability_column_name
    x_axis_title = f'{predicted_probability_column_name}'
    drift_column_name = f'{predicted_probability_column_name}_alert'
    title = f'Distribution over time for {metadata.predicted_probability_column_name}'

    fig = _joy_plot(
        feature_table=_create_feature_table(data=data),
        drift_table=drift_data,
        chunk_column_name=CHUNK_KEY_COLUMN_NAME,
        drift_column_name=drift_column_name,
        feature_column_name=predicted_probability_column_name,
        x_axis_title=x_axis_title,
        post_kde_clip=(0, 1),
        title=title,
        style='vertical',
    )
    return fig


def _create_feature_table(
    data: List[Chunk],
) -> pd.DataFrame:
    return pd.concat([chunk.data.assign(key=chunk.key) for chunk in data])
