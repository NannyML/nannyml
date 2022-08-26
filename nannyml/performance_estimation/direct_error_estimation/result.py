import pandas as pd
from plotly.graph_objects import Figure

from nannyml.base import AbstractEstimator, AbstractEstimatorResult
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_estimation.direct_error_estimation import DEFAULT_METRICS
from nannyml.plots import CHUNK_KEY_COLUMN_NAME
from nannyml.plots._step_plot import _step_plot


class Result(AbstractEstimatorResult):
    def __init__(self, results_data: pd.DataFrame, estimator: AbstractEstimator):
        super().__init__(results_data)

        from .dee import DEE

        if not isinstance(estimator, DEE):
            raise RuntimeError(
                f"{estimator.__class__.__name__} is not an instance of type " f"DataReconstructionDriftCalculator"
            )
        self.estimator = estimator

    @property
    def estimator_name(self) -> str:
        return "direct_error_estimator"

    def plot(
        self, kind: str = 'performance', metric: str = None, plot_reference: bool = False, *args, **kwargs
    ) -> Figure:
        if kind == 'performance':
            if metric is None:
                raise InvalidArgumentsException(
                    "no value for 'metric' given. Please provide the name of a metric to display."
                )
            if metric not in DEFAULT_METRICS:
                raise InvalidArgumentsException(
                    f"unknown 'metric' value: '{metric}'. " f"Supported values are {DEFAULT_METRICS}."
                )
            return _plot_direct_error_estimation_performance(self.data, metric, self.estimator, plot_reference)
        else:
            raise InvalidArgumentsException(f"unknown plot kind '{kind}'. " f"Please provide on of: ['performance'].")


def _plot_direct_error_estimation_performance(
    estimation_results: pd.DataFrame, metric: str, estimator, plot_reference: bool
) -> Figure:
    estimation_results = estimation_results.copy()

    plot_period_separator = plot_reference

    estimation_results['period'] = 'analysis'
    estimation_results['estimated'] = True

    if plot_reference:
        reference_results = estimator.previous_reference_results
        reference_results['period'] = 'reference'
        reference_results['estimated'] = False
        estimation_results = pd.concat([reference_results, estimation_results], ignore_index=True)

    # TODO: hack, assembling single results column to pass to plotting, overriding alert cols
    estimation_results['plottable'] = estimation_results.apply(
        lambda r: r[f'estimated_{metric}'] if r['period'] == 'analysis' else r[f'realized_{metric}'], axis=1
    )
    estimation_results['alert'] = estimation_results.apply(
        lambda r: r[f'alert_{metric}'] if r['period'] == 'analysis' else False, axis=1
    )

    # Plot estimated performance
    fig = _step_plot(
        table=estimation_results,
        metric_column_name='plottable',
        chunk_column_name=CHUNK_KEY_COLUMN_NAME,
        chunk_legend_labels=[f'Reference period (realized {metric})', f'Analysis period (estimated {metric})'],
        drift_column_name='alert',
        drift_legend_label='Degraded performance',
        hover_labels=['Chunk', f'{metric}', 'Target data'],
        hover_marker_labels=['Reference', 'No change', 'Change'],
        lower_threshold_column_name=f'lower_threshold_{metric}',
        upper_threshold_column_name=f'upper_threshold_{metric}',
        threshold_legend_label='Performance threshold',
        title=f'DEE - Estimated {metric}',
        y_axis_title=f'{metric}',
        v_line_separating_analysis_period=plot_period_separator,
        estimated_column_name='estimated',
        lower_confidence_column_name=f'lower_confidence_{metric}',
        upper_confidence_column_name=f'upper_confidence_{metric}',
        sampling_error_column_name=f'sampling_error_{metric}',
    )

    return fig
