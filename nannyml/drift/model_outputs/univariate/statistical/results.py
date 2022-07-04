#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing univariate statistical drift calculation results and associated plotting implementations."""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

from nannyml.base import AbstractCalculator, AbstractCalculatorResult, _column_is_categorical, _column_is_continuous
from nannyml.chunk import Chunk
from nannyml.exceptions import InvalidArgumentsException
from nannyml.plots import CHUNK_KEY_COLUMN_NAME
from nannyml.plots._joy_plot import _joy_plot
from nannyml.plots._stacked_bar_plot import _stacked_bar_plot
from nannyml.plots._step_plot import _step_plot


class UnivariateDriftResult(AbstractCalculatorResult):
    """Contains the univariate statistical drift calculation results and provides additional plotting functionality."""

    def __init__(self, results_data: pd.DataFrame, calculator: AbstractCalculator):
        super().__init__(results_data)

        from .calculator import StatisticalOutputDriftCalculator

        if not isinstance(calculator, StatisticalOutputDriftCalculator):
            raise RuntimeError(
                f"{calculator.__class__.__name__} is not an instance of type " f"UnivariateStatisticalDriftCalculator"
            )
        self.calculator = calculator

    @property
    def calculator_name(self) -> str:
        return 'univariate_statistical_output_drift'

    def plot(
        self,
        kind: str = 'prediction_drift',
        metric: str = 'statistic',
        class_label: str = None,
        plot_reference: bool = False,
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
        plot_reference: bool, default=False
            Indicates whether to include the reference period in the plot or not. Defaults to ``False``.
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
            return _plot_prediction_drift(
                self.data,
                self.calculator,
                self.calculator.y_pred,
                plot_reference,
                metric,
            )
        elif kind == 'prediction_distribution':
            return _plot_prediction_distribution(
                data=self.calculator.previous_analysis_data,
                drift_data=self.data,
                calculator=self.calculator,
                plot_reference=plot_reference,
            )
        elif kind == 'output_drift':
            return _plot_output_drift(self.data, self.calculator, plot_reference, metric, class_label)
        elif kind == 'output_distribution':
            return _plot_output_distribution(
                data=self.calculator.previous_analysis_data,
                drift_data=self.data,
                calculator=self.calculator,
                plot_reference=plot_reference,
                class_label=class_label,
            )
        else:
            raise InvalidArgumentsException(
                f"unknown plot kind '{kind}'. "
                f"Please provide on of: ['prediction_drift', 'prediction_distribution']."
            )

    # @property
    # def plots(self) -> Dict[str, go.Figure]:
    #     plots: Dict[str, go.Figure] = {}
    #
    #     if isinstance(self.metadata, BinaryClassificationMetadata):
    #         prediction_column_name = self.metadata.predicted_probability_column_name
    #         plots[f'{prediction_column_name}_drift_statistic'] = _plot_prediction_drift(
    #             self.data, self.metadata, 'statistic'
    #         )
    #         plots[f'{prediction_column_name}_drift_p_value'] = _plot_prediction_drift(
    #             self.data, self.metadata, 'p_value'
    #         )
    #         plots[f'{prediction_column_name}_distribution'] = _plot_prediction_distribution(
    #             data=self._analysis_data, drift_data=self.data, metadata=self.metadata
    #         )
    #     elif isinstance(self.metadata, MulticlassClassificationMetadata):
    #         for class_label, prediction_column_name in self.metadata.predicted_probabilities_column_names.items():
    #             plots[f'{prediction_column_name}_drift_statistic'] = _plot_prediction_drift(
    #                 self.data, self.metadata, 'statistic', class_label
    #             )
    #             plots[f'{prediction_column_name}_drift_p_value'] = _plot_prediction_drift(
    #                 self.data, self.metadata, 'p_value', class_label
    #             )
    #             plots[f'{prediction_column_name}_distribution'] = _plot_prediction_distribution(
    #                 data=self._analysis_data, drift_data=self.data, metadata=self.metadata, class_label=class_label
    #             )
    #     elif isinstance(self.metadata, RegressionMetadata):
    #         prediction_column_name = self.metadata.prediction_column_name
    #         plots[f'{prediction_column_name}_drift_statistic'] = _plot_prediction_drift(
    #             self.data, self.metadata, 'statistic'
    #         )
    #         plots[f'{prediction_column_name}_drift_p_value'] = _plot_prediction_drift(
    #             self.data, self.metadata, 'p_value'
    #         )
    #         plots[f'{prediction_column_name}_distribution'] = _plot_prediction_distribution(
    #             data=self._analysis_data, drift_data=self.data, metadata=self.metadata
    #         )
    #
    #     return plots


def _get_drift_column_names_for_feature(feature_column_name: str, feature_type: str, metric: str) -> Tuple:
    metric_column_name, metric_label, threshold_column_name = None, None, None
    if metric == 'statistic':
        if feature_type == 'categorical':
            metric_column_name = f'{feature_column_name}_chi2'
            metric_label = 'Chi-square statistic'
        elif feature_type == 'continuous':
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
    data: pd.DataFrame,
    calculator,
    y_pred: str,
    plot_reference: bool,
    metric: str = 'statistic',
) -> go.Figure:
    """Renders a line plot of the drift metric for a given feature."""
    (
        metric_column_name,
        metric_label,
        threshold_column_name,
        drift_column_name,
        title,
    ) = _get_drift_column_names_for_feature(
        y_pred,
        'continuous' if _column_is_continuous(calculator.previous_analysis_data[y_pred]) else 'categorical',
        metric,
    )

    plot_period_separator = plot_reference

    data['period'] = 'analysis'
    if plot_reference:
        reference_results = calculator.previous_reference_results
        reference_results['period'] = 'reference'
        data = pd.concat([reference_results, data], ignore_index=True)

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
    data: pd.DataFrame,
    drift_data: pd.DataFrame,
    calculator,
    plot_reference: bool,
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
    # if isinstance(metadata, BinaryClassificationMetadata):
    #     prediction_column_name = metadata.predicted_probability_column_name
    #     clip = (0, 1)
    # elif isinstance(metadata, MulticlassClassificationMetadata):
    #     if not class_label or class_label == "":
    #         raise InvalidArgumentsException("value for 'class_label' must be set when plotting for multiclass models")
    #     prediction_column_name = metadata.predicted_probabilities_column_names[class_label]
    #     clip = (0, 1)
    # elif isinstance(metadata, RegressionMetadata):
    #     prediction_column_name = metadata.prediction_column_name
    # else:
    #     raise NotImplementedError

    y_pred = calculator.y_pred
    clip = (0, 1)

    axis_title = f'{y_pred}'
    drift_column_name = f'{y_pred}_alert'
    title = f'Distribution over time for {y_pred}'

    drift_data['period'] = 'analysis'
    data['period'] = 'analysis'

    feature_table = _create_feature_table(calculator.chunker.split(data, calculator.timestamp_column_name, 'period'))

    if plot_reference:
        reference_drift = calculator.previous_reference_results
        if reference_drift is None:
            raise RuntimeError(
                f"could not plot categorical distribution for feature '{y_pred}': "
                f"calculator is missing reference results\n{calculator}"
            )
        reference_drift['period'] = 'reference'
        drift_data = pd.concat([reference_drift, drift_data], ignore_index=True)

        reference_feature_table = _create_feature_table(
            calculator.chunker.split(calculator.previous_reference_data, calculator.timestamp_column_name)
        )
        feature_table = pd.concat([reference_feature_table, feature_table], ignore_index=True)

    if _column_is_categorical(data[y_pred]):
        fig = _stacked_bar_plot(
            feature_table=feature_table,
            drift_table=drift_data,
            chunk_column_name='key',
            drift_column_name=drift_column_name,
            feature_column_name=y_pred,
            yaxis_title=axis_title,
            title=title,
        )
    elif _column_is_continuous(data[y_pred]):
        fig = _joy_plot(
            feature_table=feature_table,
            drift_table=drift_data,
            chunk_column_name=CHUNK_KEY_COLUMN_NAME,
            drift_column_name=drift_column_name,
            feature_column_name=y_pred,
            x_axis_title=axis_title,
            post_kde_clip=clip,
            title=title,
            style='vertical',
        )
    else:
        raise RuntimeError(
            f"dtype '{data[y_pred].dtype}' is not supported yet.\nPlease convert to one of "
            f"the following dtypes: ['object', 'string', 'category', 'bool'] for categorical data\n"
            f"or ['float64', 'int64'] for continuous data."
        )
    return fig


def _plot_output_drift(
    data: pd.DataFrame,
    calculator,
    plot_reference: bool,
    metric: str = 'statistic',
    class_label: str = None,
) -> go.Figure:
    """Renders a line plot of the drift metric for a given feature."""

    # deal with multiclass stuff
    if isinstance(calculator.y_pred_proba, Dict):
        if class_label is None:
            raise InvalidArgumentsException(
                "a class label is required when plotting multiclass model"
                "outputs.\nPlease provide one using the 'class_label' parameter."
            )
        if class_label not in calculator.y_pred_proba:
            raise InvalidArgumentsException(
                f"class label '{class_label}' was not found in configured "
                f"model outputs {calculator.y_pred_proba}.\n"
                f"Please provide a value that is present in the model outputs."
            )
        output_column_name = calculator.y_pred_proba[class_label]
    elif isinstance(calculator.y_pred_proba, str):
        output_column_name = calculator.y_pred_proba
    else:
        raise InvalidArgumentsException(
            "parameter 'y_pred_proba' is of type '{type(y_pred_proba)}' "
            "but should be of type 'Union[str, Dict[str, str].'"
        )

    (
        metric_column_name,
        metric_label,
        threshold_column_name,
        drift_column_name,
        title,
    ) = _get_drift_column_names_for_feature(
        output_column_name,
        'continuous' if _column_is_continuous(calculator.previous_analysis_data[output_column_name]) else 'categorical',
        metric,
    )

    plot_period_separator = plot_reference

    data['period'] = 'analysis'
    if plot_reference:
        reference_results = calculator.previous_reference_results
        reference_results['period'] = 'reference'
        data = pd.concat([reference_results, data], ignore_index=True)

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


def _plot_output_distribution(
    data: pd.DataFrame, drift_data: pd.DataFrame, calculator, plot_reference: bool, class_label: str = None
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

    # deal with multiclass stuff
    if isinstance(calculator.y_pred_proba, Dict):
        if class_label is None:
            raise InvalidArgumentsException(
                "a class label is required when plotting multiclass model"
                "outputs.\nPlease provide one using the 'class_label' parameter."
            )
        if class_label not in calculator.y_pred_proba:
            raise InvalidArgumentsException(
                f"class label '{class_label}' was not found in configured "
                f"model outputs {calculator.y_pred_proba}.\n"
                f"Please provide a value that is present in the model outputs."
            )
        output_column_name = calculator.y_pred_proba[class_label]
        clip = (0, 1)
    elif isinstance(calculator.y_pred_proba, str):
        output_column_name = calculator.y_pred_proba
        clip = (0, 1)
    else:
        raise InvalidArgumentsException(
            "parameter 'y_pred_proba' is of type '{type(y_pred_proba)}' "
            "but should be of type 'Union[str, Dict[str, str].'"
        )

    axis_title = f'{output_column_name}'
    drift_column_name = f'{output_column_name}_alert'
    title = f'Distribution over time for {output_column_name}'

    drift_data['period'] = 'analysis'
    data['period'] = 'analysis'

    feature_table = _create_feature_table(calculator.chunker.split(data, calculator.timestamp_column_name, 'period'))

    if plot_reference:
        reference_drift = calculator.previous_reference_results
        if reference_drift is None:
            raise RuntimeError(
                f"could not plot categorical distribution for feature '{output_column_name}': "
                f"calculator is missing reference results\n{calculator}"
            )
        reference_drift['period'] = 'reference'
        drift_data = pd.concat([reference_drift, drift_data], ignore_index=True)

        reference_feature_table = _create_feature_table(
            calculator.chunker.split(calculator.previous_reference_data, calculator.timestamp_column_name)
        )
        feature_table = pd.concat([reference_feature_table, feature_table], ignore_index=True)

    if _column_is_categorical(data[output_column_name]):
        fig = _stacked_bar_plot(
            feature_table=feature_table,
            drift_table=drift_data,
            chunk_column_name='key',
            drift_column_name=drift_column_name,
            feature_column_name=output_column_name,
            yaxis_title=axis_title,
            title=title,
        )
    elif _column_is_continuous(data[output_column_name]):
        fig = _joy_plot(
            feature_table=feature_table,
            drift_table=drift_data,
            chunk_column_name=CHUNK_KEY_COLUMN_NAME,
            drift_column_name=drift_column_name,
            feature_column_name=output_column_name,
            x_axis_title=axis_title,
            post_kde_clip=clip,
            title=title,
            style='vertical',
        )
    else:
        raise RuntimeError(
            f"dtype '{data[output_column_name].dtype}' is not supported yet.\nPlease convert to one of "
            f"the following dtypes: ['object', 'string', 'category', 'bool'] for categorical data\n"
            f"or ['float64', 'int64'] for continuous data."
        )
    return fig


def _create_feature_table(
    data: List[Chunk],
) -> pd.DataFrame:
    return pd.concat([chunk.data.assign(key=chunk.key) for chunk in data])
