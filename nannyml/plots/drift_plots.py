#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  #
#  License: Apache Software License 2.0\

"""Contains plots and utilities for plotting drift results."""
import pandas as pd
import plotly.graph_objects as go

from nannyml import BaseDriftCalculator, Chunker, FeatureType, InvalidArgumentsException, ModelMetadata
from nannyml.exceptions import CalculatorNotFittedException
from nannyml.plots._joy_plot import _joy_plot
from nannyml.plots._line_plot import _line_plot
from nannyml.plots._stacked_bar_plot import _stacked_bar_plot

_CHUNK_KEY_COLUMN_NAME = 'key'


class DriftPlots:
    """Utility class to generate drift result plots."""

    def __init__(self, drift_calculator: BaseDriftCalculator):
        """Creates a new instance of DriftPlots.

        Parameters
        ----------
        drift_calculator : BaseDriftCalculator
            The drift calculator used to provide the drift results. Required to provide access to `model_metadata`
            and the `chunker` that was used.

        """
        self._calculator = drift_calculator
        self._metadata = drift_calculator.model_metadata

        if drift_calculator.chunker is None:
            raise CalculatorNotFittedException(
                'the chunker for the provided calculator is not set. '
                'Please ensure your calculator was fit by using `calculator.fit()`'
            )

        self._chunker = drift_calculator.chunker

    def plot_univariate_statistical_drift(
        self,
        drift_results: pd.DataFrame,
        metric: str = 'statistic',
        feature_label: str = None,
        feature_column_name: str = None,
    ) -> go.Figure:
        """Renders a line plot for a chosen metric of univariate statistical drift calculation results.

        Given either a feature label (check ``model_metadata.features``) or the actual feature column name
        and a metric (one of either ``statistic`` or ``p_value``) this function will render a line plot displaying
        the metric value for the selected feature per chunk.
        Chunks are set on a time-based X-axis by using the period containing their observations.
        Chunks of different partitions (``reference`` and ``analysis``) are represented using different colors and
        a vertical separation if the drift results contain multiple partitions.

        Parameters
        ----------
        drift_results : pd.DataFrame
            Results of the univariate statistical drift calculation
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
        if feature_label is None and feature_column_name is None:
            raise InvalidArgumentsException("one of 'feature_label' or 'feature_column_name' should be provided.")

        feature = (
            self._metadata.feature(feature=feature_label)
            if feature_label
            else self._metadata.feature(column=feature_column_name)
        )

        if feature is None:
            raise InvalidArgumentsException(f'could not find a feature {feature_label or feature_column_name}')

        # TODO: extract helper function to improve testability
        metric_column_name, metric_label, threshold_column_name = None, None, None
        if metric == 'statistic':
            if feature.feature_type == FeatureType.CATEGORICAL:
                metric_column_name = f'{feature.column_name}_chi2'
                metric_label = 'Chi2'
            elif feature.feature_type == FeatureType.CONTINUOUS:
                metric_column_name = f'{feature.column_name}_dstat'
                metric_label = 'd-stat'
            threshold_column_name = None
        elif metric == 'p_value':
            metric_column_name = f'{feature.column_name}_p_value'
            metric_label = 'p-value'
            threshold_column_name = f'{feature.column_name}_threshold'

        plot_partition_separator = len(drift_results.value_counts()) > 1

        drift_column_name = f'{feature.column_name}_alert'
        title = f'{metric_label} evolution for {feature.label}'

        fig = _line_plot(
            table=drift_results,
            metric_column_name=metric_column_name,
            chunk_column_name=_CHUNK_KEY_COLUMN_NAME,
            drift_column_name=drift_column_name,
            threshold_column_name=threshold_column_name,
            title=title,
            y_axis_title=metric_label,
            v_line_separating_analysis_period=plot_partition_separator,
            statistically_significant_column_name=drift_column_name,
        )

        return fig

    @staticmethod
    def plot_data_reconstruction_drift(
        drift_results: pd.DataFrame,
    ) -> go.Figure:
        """Renders a line plot of the ``reconstruction_error`` of the data reconstruction drift calculation results.

        Chunks are set on a time-based X-axis by using the period containing their observations.
        Chunks of different partitions (``reference`` and ``analysis``) are represented using different colors and
        a vertical separation if the drift results contain multiple partitions.

        Parameters
        ----------
        drift_results : pd.DataFrame
            Results of the data reconstruction drift calculation

        Returns
        -------
        fig: plotly.graph_objects.Figure
            A ``Figure`` object containing the requested drift plot. Can be saved to disk or shown rendered on screen
            using ``fig.show()``.
        """
        plot_partition_separator = len(drift_results.value_counts()) > 1
        drift_results['thresholds'] = list(zip(drift_results.lower_threshold, drift_results.upper_threshold))
        fig = _line_plot(
            table=drift_results,
            metric_column_name='reconstruction_error',
            chunk_column_name=_CHUNK_KEY_COLUMN_NAME,
            drift_column_name='alert',
            threshold_column_name='thresholds',
            title='Data Reconstruction Drift',
            y_axis_title='Reconstruction Error',
            v_line_separating_analysis_period=plot_partition_separator,
        )

        return fig

    def plot_continuous_feature_distribution_over_time(
        self,
        data: pd.DataFrame,
        drift_results: pd.DataFrame,
        feature_label: str = None,
        feature_column_name: str = None,
    ) -> go.Figure:
        """Plots the data distribution and associated drift for each chunk of a given continuous feature.

        Parameters
        ----------
        data : pd.DataFrame
            The original model inputs and outputs
        drift_results : pd.DataFrame
            The results of the drift calculation
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
            A visualization of the data distribution and drift using joy-plots.
        """
        if feature_label is None and feature_column_name is None:
            raise InvalidArgumentsException("one of 'feature_label' or 'feature_column_name' should be provided.")

        feature = (
            self._metadata.feature(feature=feature_label)
            if feature_label
            else self._metadata.feature(column=feature_column_name)
        )

        if feature is None:
            raise InvalidArgumentsException(f'could not find a feature {feature_label or feature_column_name}')

        if feature.feature_type != FeatureType.CONTINUOUS:
            raise InvalidArgumentsException(
                f'cannot use features of type {repr(feature.feature_type)} in joy plots. '
                f'Please provide a feature of type {repr(FeatureType.CONTINUOUS)}'
            )

        data = data.copy(deep=True)

        feature_column_name = feature.column_name
        x_axis_title = f'{feature_column_name}'
        drift_column_name = f'{feature_column_name}_alert'
        title = f'{feature_label}: distribution over time'

        fig = _joy_plot(
            feature_table=_create_feature_table(
                data=data, chunker=self._chunker, metadata=self._metadata, feature_column_name=feature_column_name
            ),
            drift_table=drift_results,
            chunk_column_name=_CHUNK_KEY_COLUMN_NAME,
            drift_column_name=drift_column_name,
            feature_column_name=feature_column_name,
            x_axis_title=x_axis_title,
            title=title,
        )
        return fig

    def plot_categorical_feature_distribution_over_time(
        self,
        data: pd.DataFrame,
        drift_results: pd.DataFrame,
        feature_label: str = None,
        feature_column_name: str = None,
    ) -> go.Figure:
        """Plots the data distribution and associated drift for each chunk of a given categorical feature.

        Parameters
        ----------
        data : pd.DataFrame
            The original model inputs and outputs
        drift_results : pd.DataFrame
            The results of the drift calculation
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
            A visualization of the data distribution and drift using stacked bar plots.
        """
        if feature_label is None and feature_column_name is None:
            raise InvalidArgumentsException("one of 'feature_label' or 'feature_column_name' should be provided.")

        feature = (
            self._metadata.feature(feature=feature_label)
            if feature_label
            else self._metadata.feature(column=feature_column_name)
        )

        if feature is None:
            raise InvalidArgumentsException(f'could not find a feature {feature_label or feature_column_name}')

        if feature.feature_type != FeatureType.CATEGORICAL:
            raise InvalidArgumentsException(
                f'cannot use features of type {repr(feature.feature_type)} in joy plots. '
                f'Please provide a feature of type {repr(FeatureType.CATEGORICAL)}'
            )

        data = data.copy(deep=True)

        feature_column_name = feature.column_name
        x_axis_title = f'{feature_column_name}'
        drift_column_name = f'{feature_column_name}_alert'
        title = f'{feature_label}: distribution over time'

        fig = _stacked_bar_plot(
            feature_table=_create_feature_table(
                data=data, chunker=self._chunker, metadata=self._metadata, feature_column_name=feature_column_name
            ),
            drift_table=drift_results,
            chunk_column_name=_CHUNK_KEY_COLUMN_NAME,
            drift_column_name=drift_column_name,
            feature_column_name=feature_column_name,
            x_axis_title=x_axis_title,
            title=title,
        )
        return fig


class PerformancePlots:
    """Utility class to generate drift result plots."""

    def __init__(self, model_metadata: ModelMetadata, chunker: Chunker):
        """Creates a new PerformancePlots instance.

        Parameters
        ----------
        model_metadata: ModelMetadata
            The metadata used during performance estimation.
        chunker: Chunker
            The chunker used during performance estimation.
        """
        self.model_metadata = model_metadata
        self.chunker = chunker

    @staticmethod
    def plot_cbpe_performance_estimation(
        estimation_results: pd.DataFrame,
    ) -> go.Figure:
        """Renders a line plot of the ``reconstruction_error`` of the data reconstruction drift calculation results.

        Chunks are set on a time-based X-axis by using the period containing their observations.
        Chunks of different partitions (``reference`` and ``analysis``) are represented using different colors and
        a vertical separation if the drift results contain multiple partitions.

        Parameters
        ----------
        estimation_results : pd.DataFrame
            Results of the data CBPE performance estimation

        Returns
        -------
        fig: plotly.graph_objects.Figure
            A ``Figure`` object containing the requested drift plot. Can be saved to disk or shown rendered on screen
            using ``fig.show()``.
        """
        estimation_results['thresholds'] = list(
            zip(estimation_results.lower_threshold, estimation_results.upper_threshold)
        )

        estimation_results['estimated'] = True

        plot_partition_separator = len(estimation_results.value_counts()) > 1

        fig = _line_plot(
            table=estimation_results,
            metric_column_name='estimated_roc_auc',
            chunk_column_name=_CHUNK_KEY_COLUMN_NAME,
            drift_column_name='alert',
            threshold_column_name='thresholds',
            title='CBPE - estimated performance',
            y_axis_title='estimated performance',
            v_line_separating_analysis_period=plot_partition_separator,
            estimated_column_name='estimated',
            confidence_column_name='confidence',
        )

        return fig


def _create_feature_table(
    data: pd.DataFrame, metadata: ModelMetadata, chunker: Chunker, feature_column_name: str
) -> pd.DataFrame:
    data = metadata.enrich(data)
    return pd.concat([chunk.data.assign(key=chunk.key) for chunk in chunker.split(data, columns=[feature_column_name])])
