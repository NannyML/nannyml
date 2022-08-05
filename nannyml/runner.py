# #  Author:   Niels Nuyttens  <niels@nannyml.com>
# #
# #  License: Apache Software License 2.0
#
# """Used as an access point to start using NannyML in its most simple form."""
# import logging
# import sys
# from typing import List
#
import logging
from typing import Any, Dict

import pandas as pd
import rich
from rich.progress import Progress, TaskID

from nannyml.chunk import Chunker, DefaultChunker
from nannyml.drift.model_inputs.multivariate.data_reconstruction import DataReconstructionDriftCalculator
from nannyml.drift.model_inputs.univariate.statistical import UnivariateStatisticalDriftCalculator
from nannyml.drift.model_outputs.univariate.statistical import StatisticalOutputDriftCalculator
from nannyml.drift.target.target_distribution import TargetDistributionCalculator
from nannyml.io.base import Writer
from nannyml.io.file_writer import FileWriter
from nannyml.performance_calculation import PerformanceCalculator
from nannyml.performance_estimation.confidence_based import CBPE

_logger = logging.getLogger(__name__)


def run(
    reference_data: pd.DataFrame,
    analysis_data: pd.DataFrame,
    column_mapping: Dict[str, Any],
    chunker: Chunker = DefaultChunker(),
    writer: Writer = FileWriter(filepath='out', data_format='parquet'),
    continue_on_error: bool = True,
    run_in_console: bool = False,
):
    with Progress() as progress:
        _run_statistical_univariate_feature_drift_calculator(
            reference_data,
            analysis_data,
            column_mapping,
            chunker,
            writer,
            progress,
            progress.add_task('Calculate statistical univariate feature drift') if run_in_console else None,
        )

        _run_data_reconstruction_multivariate_feature_drift_calculator(
            reference_data,
            analysis_data,
            column_mapping,
            chunker,
            writer,
            progress,
            progress.add_task('Calculate data reconstruction multivariate feature drift') if run_in_console else None,
        )

        _run_statistical_model_output_drift_calculator(
            reference_data,
            analysis_data,
            column_mapping,
            chunker,
            writer,
            progress,
            progress.add_task('Calculate statistical model output drift') if run_in_console else None,
        )

        _run_target_distribution_drift_calculator(
            reference_data,
            analysis_data,
            column_mapping,
            chunker,
            writer,
            progress,
            progress.add_task('Calculate target distribution drift') if run_in_console else None,
        )

        _run_realized_performance_calculator(
            reference_data,
            analysis_data,
            column_mapping,
            chunker,
            writer,
            progress,
            progress.add_task('Calculate realized model performance') if run_in_console else None,
        )

        _run_cbpe_performance_estimation(
            reference_data,
            analysis_data,
            column_mapping,
            chunker,
            writer,
            progress,
            progress.add_task('Estimate model performance') if run_in_console else None,
        )


def _run_statistical_univariate_feature_drift_calculator(
    reference_data: pd.DataFrame,
    analysis_data: pd.DataFrame,
    column_mapping: Dict[str, Any],
    chunker: Chunker,
    writer: Writer,
    progress: rich.progress.Progress = None,
    task: rich.progress.TaskID = None,
):
    try:
        calc = UnivariateStatisticalDriftCalculator(
            feature_column_names=column_mapping['features'],
            timestamp_column_name=column_mapping['timestamp'],
            chunker=chunker,
        ).fit(reference_data)
        if progress:
            progress.update(task, advance=25)

        results = calc.calculate(analysis_data)
        raise RuntimeError("something bad happened")
        if progress:
            progress.update(task, advance=25)

        plots = {
            f'{kind}_{feature}': results.plot(kind, metric, feature)
            for feature in column_mapping['features']
            for kind in ['feature_drift', 'feature_distribution']
            for metric in ['statistic', 'p_value']
        }
        if progress:
            progress.update(task, advance=25)
    except Exception as exc:
        _logger.error(f"Failed to run statistical univariate feature drift calculator: {exc}")
        return

    writer.write(data=results.data, plots=plots, calculator_name='statistical_univariate_feature_drift')
    if progress:
        progress.update(task, advance=25)


def _run_data_reconstruction_multivariate_feature_drift_calculator(
    reference_data: pd.DataFrame,
    analysis_data: pd.DataFrame,
    column_mapping: Dict[str, Any],
    chunker: Chunker,
    writer: Writer,
    progress: rich.progress.Progress = None,
    task: rich.progress.TaskID = None,
):
    try:
        calc = DataReconstructionDriftCalculator(
            feature_column_names=column_mapping['features'],
            timestamp_column_name=column_mapping['timestamp'],
            chunker=chunker,
        ).fit(reference_data)
        if progress:
            progress.update(task, advance=25)

        results = calc.calculate(analysis_data)
        if progress:
            progress.update(task, advance=25)

        plots = {f'{kind}': results.plot(kind='drift') for kind in ['drift']}
        if progress:
            progress.update(task, advance=25)
    except Exception as exc:
        _logger.error(f"Failed to run data reconstruction multivariate feature calculator: {exc}")
        return

    writer.write(data=results.data, plots=plots, calculator_name='data_reconstruction_multivariate_feature_drift')
    if progress:
        progress.update(task, advance=25)


def _run_statistical_model_output_drift_calculator(
    reference_data: pd.DataFrame,
    analysis_data: pd.DataFrame,
    column_mapping: Dict[str, Any],
    chunker: Chunker,
    writer: Writer,
    progress: rich.progress.Progress = None,
    task: TaskID = None,
):
    try:
        calc = StatisticalOutputDriftCalculator(
            y_pred=column_mapping['y_pred'],
            y_pred_proba=column_mapping['y_pred_proba'],
            timestamp_column_name=column_mapping['timestamp'],
            chunker=chunker,
        ).fit(reference_data)
        if progress:
            progress.update(task, advance=25)

        results = calc.calculate(analysis_data)
        if progress:
            progress.update(task, advance=25)

        is_multiclass = isinstance(column_mapping['y_pred_proba'], dict)
        if is_multiclass:
            classes = list(column_mapping['y_pred_proba'].keys())
            plots = {
                f'{kind}_{metric}_{clazz}': results.plot(kind, metric, class_label=clazz)
                for kind in [
                    'predicted_labels_drift',
                    'predicted_labels_distribution',
                    'prediction_drift',
                    'prediction_distribution',
                ]
                for metric in ['statistic', 'p_value']
                for clazz in classes
            }
        else:
            plots = {
                f'{kind}_{metric}': results.plot(kind, metric)
                for kind in ['predicted_labels_drift', 'prediction_drift']
                for metric in ['statistic', 'p_value']
            }
            plots.update(
                {f'{kind}': results.plot(kind) for kind in ['predicted_labels_distribution', 'prediction_distribution']}
            )
        if progress:
            progress.update(task, advance=25)
    except Exception as exc:
        _logger.error(f"Failed to run model output drift calculator: {exc}")
        return

    writer.write(data=results.data, plots=plots, calculator_name='statistical_model_output_drift')
    if progress:
        progress.update(task, advance=25)


def _run_target_distribution_drift_calculator(
    reference_data: pd.DataFrame,
    analysis_data: pd.DataFrame,
    column_mapping: Dict[str, Any],
    chunker: Chunker,
    writer: Writer,
    progress: rich.progress.Progress = None,
    task: TaskID = None,
):
    if column_mapping['y_true'] not in analysis_data.columns:
        _logger.info(
            f"target values column '{column_mapping['y_true']}' not present in analysis data. "
            "Skipping target distribution calculation."
        )
        if progress:
            rich.print(
                f"Target values column '{column_mapping['y_true']}' not present in analysis data. "
                "Skipping target distribution calculation."
            )
        return

    try:
        calc = TargetDistributionCalculator(
            y_true=column_mapping['y_true'], timestamp_column_name=column_mapping['timestamp'], chunker=chunker
        ).fit(reference_data)
        if progress:
            progress.update(task, advance=25)

        results = calc.calculate(analysis_data)
        if progress:
            progress.update(task, advance=25)

        plots = {
            f'{kind}_{distribution}': results.plot(kind, distribution)
            for kind in ['distribution']
            for distribution in ['statistical', 'metric']
        }
        if progress:
            progress.update(task, advance=25)
    except Exception as exc:
        _logger.error(f"Failed to run target distribution calculator: {exc}")
        return

    writer.write(data=results.data, plots=plots, calculator_name='target_distribution')
    if progress:
        progress.update(task, advance=25)


def _run_realized_performance_calculator(
    reference_data: pd.DataFrame,
    analysis_data: pd.DataFrame,
    column_mapping: Dict[str, Any],
    chunker: Chunker,
    writer: Writer,
    progress: rich.progress.Progress = None,
    task: TaskID = None,
):
    if column_mapping['y_true'] not in analysis_data.columns:
        _logger.info(
            f"target values column '{column_mapping['y_true']}' not present in analysis data. "
            "Skipping realized performance calculation."
        )
        if progress:
            rich.print(
                f"Target values column '{column_mapping['y_true']}' not present in analysis data. "
                "Skipping realized performance calculation."
            )
        return

    metrics = ['roc_auc', 'f1', 'precision', 'recall', 'specificity', 'accuracy']

    try:
        calc = PerformanceCalculator(
            y_true=column_mapping['y_true'],
            y_pred=column_mapping['y_pred'],
            y_pred_proba=column_mapping['y_pred_proba'],
            timestamp_column_name=column_mapping['timestamp'],
            chunker=chunker,
            metrics=metrics,
        ).fit(reference_data)
        if progress:
            progress.update(task, advance=25)

        results = calc.calculate(analysis_data)
        if progress:
            progress.update(task, advance=25)

        plots = {
            f'realized_{metric}': results.plot(kind, metric=metric) for kind in ['performance'] for metric in metrics
        }
        if progress:
            progress.update(task, advance=25)
    except Exception as exc:
        _logger.error(f"Failed to run realized performance calculator: {exc}")
        return

    writer.write(data=results.data, plots=plots, calculator_name='realized_performance')
    if progress:
        progress.update(task, advance=25)


def _run_cbpe_performance_estimation(
    reference_data: pd.DataFrame,
    analysis_data: pd.DataFrame,
    column_mapping: Dict[str, Any],
    chunker: Chunker,
    writer: Writer,
    progress: rich.progress.Progress = None,
    task: TaskID = None,
):
    metrics = ['roc_auc', 'f1', 'precision', 'recall', 'specificity', 'accuracy']

    try:
        estimator = CBPE(  # type: ignore
            y_true=column_mapping['y_true'],
            y_pred=column_mapping['y_pred'],
            y_pred_proba=column_mapping['y_pred_proba'],
            timestamp_column_name=column_mapping['timestamp'],
            chunker=chunker,
            metrics=metrics,
        ).fit(reference_data)
        if progress:
            progress.update(task, advance=25)

        results = estimator.estimate(analysis_data)
        if progress:
            progress.update(task, advance=25)

        plots = {
            f'estimated_{metric}': results.plot(kind, metric=metric) for kind in ['performance'] for metric in metrics
        }
        if progress:
            progress.update(task, advance=25)

    except Exception as exc:
        _logger.error(f"Failed to run CBPE performance estimator: {exc}")
        return

    writer.write(data=results.data, plots=plots, calculator_name='estimated_performance')
    if progress:
        progress.update(task, advance=25)
