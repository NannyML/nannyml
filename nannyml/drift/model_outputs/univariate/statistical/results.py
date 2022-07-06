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

"""Contains the results of the model output statistical drift calculation and provides plotting functionality."""


class UnivariateDriftResult(AbstractCalculatorResult):
    """Contains the results of the model output statistical drift calculation and provides plotting functionality."""

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
        """Renders plots for metrics returned by the univariate statistical drift calculator.

        For both model predictions and outputs you can render the statistic value or p-values as a step plot,
        or create a distribution plot. For multiclass use cases it is required to provide a ``class_label`` parameter
        when rendering model output plots.

        Select a plot using the ``kind`` parameter:

        - ``predicted_labels_drift``
                plots the drift metric per :class:`~nannyml.chunk.Chunk` for the model predictions ``y_pred``.
        - ``predicted_labels_distribution``
                plots the distribution per :class:`~nannyml.chunk.Chunk` for the model predictions ``y_pred``.
        - ``prediction_drift``
                plots the drift metric per :class:`~nannyml.chunk.Chunk` for the model outputs ``y_pred_proba``.
        - ``prediction_distribution``
                plots the distribution per per :class:`~nannyml.chunk.Chunk` for the model outputs ``y_pred_proba``


        Parameters
        ----------
        kind: str, default=`predicted_labels_drift`
            The kind of plot you want to have. Allowed values are ``predicted_labels_drift``,
            ``predicted_labels_distribution``, ``prediction_drift`` and ``prediction_distribution``.
        metric : str, default=``statistic``
            The metric to plot. Allowed values are ``statistic`` and ``p_value``.
            Not applicable when plotting distributions.
        plot_reference: bool, default=False
            Indicates whether to include the reference period in the plot or not. Defaults to ``False``.
        class_label: str, default=None
            The label of the class to plot the prediction distribution for.
            Only required in case of multiclass use cases.


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
        >>> calc = nml.StatisticalOutputDriftCalculator(
        >>>     y_pred_proba='y_pred_proba',
        >>>     y_pred='y_pred',
        >>>     timestamp_column_name='timestamp'
        >>> )
        >>> calc.fit(reference_df)
        >>> results = calc.calculate(analysis_df)
        >>>
        >>> print(results.data)  # check the numbers
                     key  start_index  ...  y_pred_proba_alert y_pred_proba_threshold
        0       [0:4999]            0  ...                True                   0.05
        1    [5000:9999]         5000  ...               False                   0.05
        2  [10000:14999]        10000  ...               False                   0.05
        3  [15000:19999]        15000  ...               False                   0.05
        4  [20000:24999]        20000  ...               False                   0.05
        5  [25000:29999]        25000  ...                True                   0.05
        6  [30000:34999]        30000  ...                True                   0.05
        7  [35000:39999]        35000  ...                True                   0.05
        8  [40000:44999]        40000  ...                True                   0.05
        9  [45000:49999]        45000  ...                True                   0.05
        >>>
        >>> results.plot(kind='predicted_labels_drift', metric='p_value', plot_reference=True).show()
        >>> results.plot(kind='predicted_labels_distribution', plot_reference=True).show()
        >>> results.plot(kind='prediction_drift', plot_reference=True).show()
        >>> results.plot(kind='prediction_distribution', plot_reference=True).show()

        """
        if kind == 'predicted_labels_drift':
            return _plot_prediction_drift(
                self.data,
                self.calculator,
                self.calculator.y_pred,
                plot_reference,
                metric,
            )
        elif kind == 'predicted_labels_distribution':
            return _plot_prediction_distribution(
                data=self.calculator.previous_analysis_data,
                drift_data=self.data,
                calculator=self.calculator,
                plot_reference=plot_reference,
            )
        elif kind == 'prediction_drift':
            return _plot_output_drift(self.data, self.calculator, plot_reference, metric, class_label)
        elif kind == 'prediction_distribution':
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
                "Please provide on of: ['prediction_drift', 'prediction_distribution', 'predicted_labels_drift',"
                "'predicted_labels_distribution']."
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
    y_pred = calculator.y_pred
    axis_title = f'{y_pred}'
    drift_column_name = f'{y_pred}_alert'
    title = f'Distribution over time for {y_pred}'

    drift_data['period'] = 'analysis'
    data['period'] = 'analysis'

    feature_table = _create_feature_table(calculator.chunker.split(data, calculator.timestamp_column_name))

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

    fig = _stacked_bar_plot(
        feature_table=feature_table,
        drift_table=drift_data,
        chunk_column_name='key',
        drift_column_name=drift_column_name,
        feature_column_name=y_pred,
        yaxis_title=axis_title,
        title=title,
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

    feature_table = _create_feature_table(calculator.chunker.split(data, calculator.timestamp_column_name))

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
