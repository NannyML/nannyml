#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing univariate statistical drift calculation results and associated plotting implementations."""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

from nannyml.chunk import Chunk
from nannyml.drift.base import DriftResult
from nannyml.exceptions import InvalidArgumentsException
from nannyml.metadata import BinaryClassificationMetadata, MulticlassClassificationMetadata, RegressionMetadata
from nannyml.metadata.base import Feature, FeatureType, ModelMetadata
from nannyml.plots import CHUNK_KEY_COLUMN_NAME
from nannyml.plots._joy_plot import _joy_plot
from nannyml.plots._step_plot import _step_plot


class UnivariateDriftResult(DriftResult):
    """Contains the univariate statistical drift calculation results and provides additional plotting functionality."""

    calculator_name: str = 'univariate_statistical_output_drift'

    # TODO: this is messing up functionality in scratch files (sets runtime class to DataFrame). Check this!
    def __repr__(self):
        """Represent the DriftResults object as the data it contains."""
        return self.data.__repr__()

    def plot(
        self,
        kind: str = 'prediction_drift',
        metric: str = 'statistic',
        class_label: str = None,
        *args,
        **kwargs,
    ) -> go.Figure:
        """Renders a line plot for a chosen metric of statistical statistical drift calculation results.

        Given either a feature label (check ``model_metadata.features``) or the actual feature column name
        and a metric (one of either ``statistic`` or ``p_value``) this function will render a line plot displaying
        the metric value for the selected feature per chunk.
        Chunks are set on a time-based X-axis by using the period containing their observations.
        Chunks of different periods (``reference`` and ``analysis``) are represented using different colors and
        a vertical separation if the drift results contain multiple periods.

        The different plot kinds that are available:

        - ``prediction_drift``: plots drift per :class:`~nannyml.chunk.Chunk` for the predictions of a chunked data set.
        - ``prediction_distribution``: plots the prediction distribution per :class:`~nannyml.chunk.Chunk` of a chunked
          data set as a joyplot.


        Parameters
        ----------
        kind: str, default=`prediction_drift`
            The kind of plot you want to have. Value must be one of ``prediction_drift``, ``prediction_distribution``.
        metric : str, default=``statistic``
            The metric to plot. Value must be one of ``statistic`` or ``p_value``
        class_label: str, default=None
            The label of the class to plot the prediction distribution for. Only required in case of multiclass models.


        Returns
        -------
        fig: plotly.graph_objects.Figure
            A ``Figure`` object containing the requested drift plot. Can be saved to disk or shown rendered on screen
            using ``fig.show()``.


        Examples
        --------
        >>> import nannyml as nml
        >>> ref_df, ana_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>> metadata = nml.extract_metadata(ref_df, model_type=nml.ModelType.CLASSIFICATION_BINARY)
        >>> drift_calc = nml.UnivariateStatisticalDriftCalculator(model_metadata=metadata, chunk_period='W')
        >>> drift_calc.fit(ref_df)
        >>> drifts = drift_calc.calculate(ana_df)
        >>> # loop over all features and plot the feature drift and feature distribution for each
        >>> for f in metadata.features:
        >>>     drifts.plot(kind='feature_drift', feature_label=f.label).show()
        >>>     drifts.plot(kind='feature_distribution', feature_label=f.label).show()

        """
        if kind == 'prediction_drift':
            return _plot_prediction_drift(self.data, self.metadata, metric, class_label)
        elif kind == 'prediction_distribution':
            return _plot_prediction_distribution(
                data=self._analysis_data, drift_data=self.data, metadata=self.metadata, class_label=class_label
            )
        else:
            raise InvalidArgumentsException(
                f"unknown plot kind '{kind}'. "
                f"Please provide on of: ['prediction_drift', 'prediction_distribution']."
            )

    @property
    def plots(self) -> Dict[str, go.Figure]:
        plots: Dict[str, go.Figure] = {}

        if isinstance(self.metadata, BinaryClassificationMetadata):
            prediction_column_name = self.metadata.predicted_probability_column_name
            plots[f'{prediction_column_name}_drift_statistic'] = _plot_prediction_drift(
                self.data, self.metadata, 'statistic'
            )
            plots[f'{prediction_column_name}_drift_p_value'] = _plot_prediction_drift(
                self.data, self.metadata, 'p_value'
            )
            plots[f'{prediction_column_name}_distribution'] = _plot_prediction_distribution(
                data=self._analysis_data, drift_data=self.data, metadata=self.metadata
            )
        elif isinstance(self.metadata, MulticlassClassificationMetadata):
            for class_label, prediction_column_name in self.metadata.predicted_probabilities_column_names.items():
                plots[f'{prediction_column_name}_drift_statistic'] = _plot_prediction_drift(
                    self.data, self.metadata, 'statistic', class_label
                )
                plots[f'{prediction_column_name}_drift_p_value'] = _plot_prediction_drift(
                    self.data, self.metadata, 'p_value', class_label
                )
                plots[f'{prediction_column_name}_distribution'] = _plot_prediction_distribution(
                    data=self._analysis_data, drift_data=self.data, metadata=self.metadata, class_label=class_label
                )
        elif isinstance(self.metadata, RegressionMetadata):
            prediction_column_name = self.metadata.prediction_column_name
            plots[f'{prediction_column_name}_drift_statistic'] = _plot_prediction_drift(
                self.data, self.metadata, 'statistic'
            )
            plots[f'{prediction_column_name}_drift_p_value'] = _plot_prediction_drift(
                self.data, self.metadata, 'p_value'
            )
            plots[f'{prediction_column_name}_distribution'] = _plot_prediction_distribution(
                data=self._analysis_data, drift_data=self.data, metadata=self.metadata
            )

        return plots


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


def _plot_prediction_drift(
    data: pd.DataFrame, metadata: ModelMetadata, metric: str = 'statistic', class_label: str = None
) -> go.Figure:
    """Renders a line plot of the drift metric for a given feature."""
    if isinstance(metadata, BinaryClassificationMetadata):
        prediction_column_name = metadata.predicted_probability_column_name
    elif isinstance(metadata, MulticlassClassificationMetadata):
        if not class_label or class_label == "":
            raise InvalidArgumentsException("value for 'class_label' must be set when plotting for multiclass models")
        if class_label not in metadata.predicted_probabilities_column_names:
            raise InvalidArgumentsException(f"no classes found named '{class_label}'. Please review the given value.")
        prediction_column_name = metadata.predicted_probabilities_column_names[class_label]
    elif isinstance(metadata, RegressionMetadata):
        prediction_column_name = metadata.prediction_column_name
    else:
        raise NotImplementedError

    (
        metric_column_name,
        metric_label,
        threshold_column_name,
        drift_column_name,
        title,
    ) = _get_drift_column_names_for_feature(prediction_column_name, FeatureType.CONTINUOUS, metric)

    plot_period_separator = len(data.value_counts()) > 1

    fig = _step_plot(
        table=data,
        metric_column_name=metric_column_name,
        chunk_column_name=CHUNK_KEY_COLUMN_NAME,
        drift_column_name=drift_column_name,
        lower_threshold_column_name=threshold_column_name,
        hover_labels=['Chunk', metric_label, 'Target data'],
        title=title,
        y_axis_title=metric_label,
        v_line_separating_analysis_period=plot_period_separator,
        statistically_significant_column_name=drift_column_name,
    )
    return fig


def _plot_prediction_distribution(
    data: List[Chunk], drift_data: pd.DataFrame, metadata: ModelMetadata, class_label: str = None
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
    class_label: str, default=None
        The label of the class to plot the prediction distribution for. Only required in case of multiclass models.

    Returns
    -------
    fig: plotly.graph_objects.Figure
        A visualization of the data distribution and drift using joy-plots.
    """
    clip: Optional[Tuple[int, int]] = None
    if isinstance(metadata, BinaryClassificationMetadata):
        prediction_column_name = metadata.predicted_probability_column_name
        clip = (0, 1)
    elif isinstance(metadata, MulticlassClassificationMetadata):
        if not class_label or class_label == "":
            raise InvalidArgumentsException("value for 'class_label' must be set when plotting for multiclass models")
        prediction_column_name = metadata.predicted_probabilities_column_names[class_label]
        clip = (0, 1)
    elif isinstance(metadata, RegressionMetadata):
        prediction_column_name = metadata.prediction_column_name
    else:
        raise NotImplementedError

    x_axis_title = f'{prediction_column_name}'
    drift_column_name = f'{prediction_column_name}_alert'
    title = f'Distribution over time for {prediction_column_name}'

    fig = _joy_plot(
        feature_table=_create_feature_table(data=data),
        drift_table=drift_data,
        chunk_column_name=CHUNK_KEY_COLUMN_NAME,
        drift_column_name=drift_column_name,
        feature_column_name=prediction_column_name,
        x_axis_title=x_axis_title,
        post_kde_clip=clip,
        title=title,
        style='vertical',
    )
    return fig


def _create_feature_table(
    data: List[Chunk],
) -> pd.DataFrame:
    return pd.concat([chunk.data.assign(key=chunk.key) for chunk in data])
