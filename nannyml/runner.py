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
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

from nannyml.chunk import Chunker
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
    chunker: Chunker,
    writer: Writer = FileWriter(filepath='out', data_format='parquet'),
    ignore_errors: bool = True,
    run_in_console: bool = False,
):
    with Progress() as progress:
        task = progress.add_task('Calculating drift', total=1) if run_in_console else None
        _run_statistical_univariate_feature_drift_calculator(
            reference_data, analysis_data, column_mapping, chunker, writer, ignore_errors, console=progress.console
        )
        if task is not None:
            progress.update(task, advance=1 / 6)

        _run_data_reconstruction_multivariate_feature_drift_calculator(
            reference_data, analysis_data, column_mapping, chunker, writer, ignore_errors, console=progress.console
        )
        if task is not None:
            progress.update(task, advance=2 / 6)

        _run_statistical_model_output_drift_calculator(
            reference_data, analysis_data, column_mapping, chunker, writer, ignore_errors, console=progress.console
        )
        if task is not None:
            progress.update(task, advance=3 / 6)

        _run_target_distribution_drift_calculator(
            reference_data, analysis_data, column_mapping, chunker, writer, ignore_errors, console=progress.console
        )
        if task is not None:
            progress.update(task, advance=4 / 6)

        _run_realized_performance_calculator(
            reference_data, analysis_data, column_mapping, chunker, writer, ignore_errors, console=progress.console
        )
        if task is not None:
            progress.update(task, description='Calculating realized performance', advance=5 / 6)

        _run_cbpe_performance_estimation(
            reference_data, analysis_data, column_mapping, chunker, writer, ignore_errors, console=progress.console
        )
        if task is not None:
            progress.update(task, description='Estimating performance', advance=6 / 6)

        progress.console.line(2)
        progress.console.print(Panel(f"View results in {Path(writer.filepath)}"))


def _run_statistical_univariate_feature_drift_calculator(
    reference_data: pd.DataFrame,
    analysis_data: pd.DataFrame,
    column_mapping: Dict[str, Any],
    chunker: Chunker,
    writer: Writer,
    ignore_errors: bool,
    console: Console = None,
):
    if console:
        console.rule('[cyan]UnivariateStatisticalDriftCalculator[/]')
    try:
        if console:
            console.log('fitting on reference data')
        calc = UnivariateStatisticalDriftCalculator(
            feature_column_names=column_mapping['features'],
            timestamp_column_name=column_mapping['timestamp'],
            chunker=chunker,
        ).fit(reference_data)

        # raise RuntimeError("🔥 something's not right there... 🔥")

        if console:
            console.log('calculating on analysis data')
        results = calc.calculate(analysis_data)

        if console:
            console.log('generating result plots')
        plots = {
            f'{kind}_{feature}': results.plot(kind, metric, feature)
            for feature in column_mapping['features']
            for kind in ['feature_drift', 'feature_distribution']
            for metric in ['statistic', 'p_value']
        }
    except Exception as exc:
        msg = f"Failed to run statistical univariate feature drift calculator: {exc}"
        if console:
            console.log(msg, style='red')
        else:
            _logger.error(msg)
        if ignore_errors:
            return
        else:
            sys.exit(1)

    if console:
        console.log('writing results')
    writer.write(data=results.data, plots=plots, calculator_name='statistical_univariate_feature_drift')


def _run_data_reconstruction_multivariate_feature_drift_calculator(
    reference_data: pd.DataFrame,
    analysis_data: pd.DataFrame,
    column_mapping: Dict[str, Any],
    chunker: Chunker,
    writer: Writer,
    ignore_errors: bool,
    console: Console = None,
):
    if console:
        console.rule('[cyan]DataReconstructionDriftCalculator[/]')
    try:
        if console:
            console.log('fitting on reference data')
        calc = DataReconstructionDriftCalculator(
            feature_column_names=column_mapping['features'],
            timestamp_column_name=column_mapping['timestamp'],
            chunker=chunker,
        ).fit(reference_data)

        if console:
            console.log('calculating on analysis data')
        results = calc.calculate(analysis_data)

        if console:
            console.log('generating result plots')
        plots = {f'{kind}': results.plot(kind='drift') for kind in ['drift']}
    except Exception as exc:
        msg = f"Failed to run data reconstruction multivariate feature drift calculator: {exc}"
        if console:
            console.log(msg, style='red')
        else:
            _logger.error(msg)
        if ignore_errors:
            return
        else:
            sys.exit(1)

    if console:
        console.log('writing results')
    writer.write(data=results.data, plots=plots, calculator_name='data_reconstruction_multivariate_feature_drift')


def _run_statistical_model_output_drift_calculator(
    reference_data: pd.DataFrame,
    analysis_data: pd.DataFrame,
    column_mapping: Dict[str, Any],
    chunker: Chunker,
    writer: Writer,
    ignore_errors: bool,
    console: Console = None,
):
    if console:
        console.rule('[cyan]UnivariateStatisticalDriftCalculator[/]')
    try:
        if console:
            console.log('fitting on reference data')
        calc = StatisticalOutputDriftCalculator(
            y_pred=column_mapping['y_pred'],
            y_pred_proba=column_mapping['y_pred_proba'],
            timestamp_column_name=column_mapping['timestamp'],
            chunker=chunker,
        ).fit(reference_data)

        if console:
            console.log('calculating on analysis data')
        results = calc.calculate(analysis_data)

        if console:
            console.log('generating result plots')
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
    except Exception as exc:
        msg = f"Failed to run model output drift calculator: {exc}"
        if console:
            console.log(msg, style='red')
        else:
            _logger.error(msg)
        if ignore_errors:
            return
        else:
            sys.exit(1)

    if console:
        console.log('writing results')
    writer.write(data=results.data, plots=plots, calculator_name='statistical_model_output_drift')


def _run_target_distribution_drift_calculator(
    reference_data: pd.DataFrame,
    analysis_data: pd.DataFrame,
    column_mapping: Dict[str, Any],
    chunker: Chunker,
    writer: Writer,
    ignore_errors: bool,
    console: Console = None,
):
    if console:
        console.rule('[cyan]TargetDistributionCalculator[/]')

    if column_mapping['y_true'] not in analysis_data.columns:
        _logger.info(
            f"target values column '{column_mapping['y_true']}' not present in analysis data. "
            "Skipping target distribution calculation."
        )
        if console:
            console.log(
                f"target values column '{column_mapping['y_true']}' not present in analysis data. "
                "Skipping target distribution calculation.",
                style='yellow',
            )
        return

    try:
        if console:
            console.log('fitting on reference data')
        calc = TargetDistributionCalculator(
            y_true=column_mapping['y_true'], timestamp_column_name=column_mapping['timestamp'], chunker=chunker
        ).fit(reference_data)

        if console:
            console.log('calculating on analysis data')
        results = calc.calculate(analysis_data)

        if console:
            console.log('generating result plots')
        plots = {
            f'{kind}_{distribution}': results.plot(kind, distribution)
            for kind in ['distribution']
            for distribution in ['statistical', 'metric']
        }
    except Exception as exc:
        msg = f"Failed to run target distribution calculator: {exc}"
        if console:
            console.log(msg, style='red')
        else:
            _logger.error(msg)
        if ignore_errors:
            return
        else:
            sys.exit(1)

    if console:
        console.log('writing results')
    writer.write(data=results.data, plots=plots, calculator_name='target_distribution')


def _run_realized_performance_calculator(
    reference_data: pd.DataFrame,
    analysis_data: pd.DataFrame,
    column_mapping: Dict[str, Any],
    chunker: Chunker,
    writer: Writer,
    ignore_errors: bool,
    console: Console = None,
):
    if console:
        console.rule('[cyan]PerformanceCalculator[/]')

    if column_mapping['y_true'] not in analysis_data.columns:
        _logger.info(
            f"target values column '{column_mapping['y_true']}' not present in analysis data. "
            "Skipping realized performance calculation."
        )
        if console:
            console.log(
                f"target values column '{column_mapping['y_true']}' not present in analysis data. "
                "Skipping realized performance calculation.",
                style='yellow',
            )
        return

    metrics = ['roc_auc', 'f1', 'precision', 'recall', 'specificity', 'accuracy']

    try:
        if console:
            console.log('fitting on reference data')
        calc = PerformanceCalculator(
            y_true=column_mapping['y_true'],
            y_pred=column_mapping['y_pred'],
            y_pred_proba=column_mapping['y_pred_proba'],
            timestamp_column_name=column_mapping['timestamp'],
            chunker=chunker,
            metrics=metrics,
        ).fit(reference_data)

        if console:
            console.log('calculating on analysis data')
        results = calc.calculate(analysis_data)

        if console:
            console.log('generating result plots')
        plots = {
            f'realized_{metric}': results.plot(kind, metric=metric) for kind in ['performance'] for metric in metrics
        }
    except Exception as exc:
        msg = f"Failed to run realized performance calculator: {exc}"
        if console:
            console.log(msg, style='red')
        else:
            _logger.error(msg)
        if ignore_errors:
            return
        else:
            sys.exit(1)

    if console:
        console.log('writing results')
    writer.write(data=results.data, plots=plots, calculator_name='realized_performance')


def _run_cbpe_performance_estimation(
    reference_data: pd.DataFrame,
    analysis_data: pd.DataFrame,
    column_mapping: Dict[str, Any],
    chunker: Chunker,
    writer: Writer,
    ignore_errors: bool,
    console: Console = None,
):
    metrics = ['roc_auc', 'f1', 'precision', 'recall', 'specificity', 'accuracy']

    if console:
        console.rule('[cyan]PerformanceEstimator[/]')

    try:
        if console:
            console.log('fitting on reference data')
        estimator = CBPE(  # type: ignore
            y_true=column_mapping['y_true'],
            y_pred=column_mapping['y_pred'],
            y_pred_proba=column_mapping['y_pred_proba'],
            timestamp_column_name=column_mapping['timestamp'],
            chunker=chunker,
            metrics=metrics,
        ).fit(reference_data)
        if console:
            console.log('estimating on analysis data')
        results = estimator.estimate(analysis_data)

        if console:
            console.log('generating result plots')
        plots = {
            f'estimated_{metric}': results.plot(kind, metric=metric) for kind in ['performance'] for metric in metrics
        }

    except Exception as exc:
        msg = f"Failed to run CBPE performance estimator: {exc}"
        if console:
            console.log(msg, style='red')
        else:
            _logger.error(msg)
        if ignore_errors:
            return
        else:
            sys.exit(1)

    if console:
        console.log('writing results')
    writer.write(data=results.data, plots=plots, calculator_name='estimated_performance')
