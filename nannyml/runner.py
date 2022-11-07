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
from rich.progress import Progress

from nannyml._typing import ProblemType
from nannyml.chunk import Chunker
from nannyml.drift.multivariate.data_reconstruction import DataReconstructionDriftCalculator
from nannyml.drift.univariate import UnivariateDriftCalculator
from nannyml.io.base import Writer
from nannyml.io.raw_files_writer import RawFilesWriter
from nannyml.performance_calculation import PerformanceCalculator
from nannyml.performance_estimation.confidence_based import CBPE
from nannyml.performance_estimation.direct_loss_estimation import DEFAULT_METRICS, DLE

_logger = logging.getLogger(__name__)


def run(
    reference_data: pd.DataFrame,
    analysis_data: pd.DataFrame,
    column_mapping: Dict[str, Any],
    problem_type: ProblemType,
    chunker: Chunker,
    writer: Writer,
    ignore_errors: bool = True,
    run_in_console: bool = False,
):
    with Progress() as progress:
        _run_statistical_univariate_feature_drift_calculator(
            reference_data,
            analysis_data,
            column_mapping,
            problem_type,
            chunker,
            writer,
            ignore_errors,
            console=progress.console,
        )

        _run_data_reconstruction_multivariate_feature_drift_calculator(
            reference_data, analysis_data, column_mapping, chunker, writer, ignore_errors, console=progress.console
        )

        _run_realized_performance_calculator(
            reference_data,
            analysis_data,
            column_mapping,
            problem_type,
            chunker,
            writer,
            ignore_errors,
            console=progress.console,
        )

        _run_cbpe_performance_estimation(
            reference_data,
            analysis_data,
            column_mapping,
            problem_type,
            chunker,
            writer,
            ignore_errors,
            console=progress.console,
        )

        _run_dee_performance_estimation(
            reference_data,
            analysis_data,
            column_mapping,
            problem_type,
            chunker,
            writer,
            ignore_errors,
            console=progress.console,
        )

        progress.console.line(2)
        if isinstance(writer, RawFilesWriter):
            progress.console.rule()
            progress.console.log(f"view results in {Path(writer.filepath)}")


def _run_statistical_univariate_feature_drift_calculator(
    reference_data: pd.DataFrame,
    analysis_data: pd.DataFrame,
    column_mapping: Dict[str, Any],
    problem_type: ProblemType,
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
        if problem_type == ProblemType.CLASSIFICATION_BINARY:
            y_pred_proba_column_names = [column_mapping['y_pred_proba']]
        elif problem_type == ProblemType.CLASSIFICATION_MULTICLASS:
            y_pred_proba_column_names = list(column_mapping['y_pred_proba'].values())
        else:
            y_pred_proba_column_names = []

        calc = UnivariateDriftCalculator(
            column_names=(column_mapping['features'] + [column_mapping['y_pred']] + y_pred_proba_column_names),
            timestamp_column_name=column_mapping.get('timestamp', None),
            chunker=chunker,
            categorical_methods=['chi2', 'jensen_shannon'],
            continuous_methods=['kolmogorov_smirnov', 'jensen_shannon'],
        ).fit(reference_data)

        # raise RuntimeError("🔥 something's not right there... 🔥")

        if console:
            console.log('calculating on analysis data')
        results = calc.calculate(analysis_data)

        plots = {}
        if isinstance(writer, RawFilesWriter):
            if console:
                console.log('generating result plots')
            plots = {
                f'{kind}_{column_name}': results.plot(kind=kind, method=method, column_name=column_name)
                for column_name in results.continuous_column_names
                for method in results.continuous_method_names
                for kind in ['drift', 'distribution']
            }
            plots.update(
                {
                    f'{kind}_{column_name}': results.plot(kind=kind, method=method, column_name=column_name)
                    for column_name in results.categorical_column_names
                    for method in results.categorical_method_names
                    for kind in ['drift', 'distribution']
                }
            )
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
    writer.write(result=results, plots=plots, calculator_name='statistical_univariate_feature_drift')


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
            column_names=column_mapping['features'],
            timestamp_column_name=column_mapping.get('timestamp', None),
            chunker=chunker,
        ).fit(reference_data)

        if console:
            console.log('calculating on analysis data')
        results = calc.calculate(analysis_data)

        plots = {}
        if isinstance(writer, RawFilesWriter):
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
    writer.write(result=results, plots=plots, calculator_name='data_reconstruction_multivariate_feature_drift')


def _run_realized_performance_calculator(
    reference_data: pd.DataFrame,
    analysis_data: pd.DataFrame,
    column_mapping: Dict[str, Any],
    problem_type: ProblemType,
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

    metrics = []
    if problem_type in [ProblemType.CLASSIFICATION_BINARY, ProblemType.CLASSIFICATION_MULTICLASS]:
        metrics = ['roc_auc', 'f1', 'precision', 'recall', 'specificity', 'accuracy']
    elif problem_type in [ProblemType.REGRESSION]:
        metrics = DEFAULT_METRICS

    try:
        if console:
            console.log('fitting on reference data')
        calc = PerformanceCalculator(
            y_true=column_mapping['y_true'],
            y_pred=column_mapping['y_pred'],
            y_pred_proba=column_mapping.get('y_pred_proba', None),
            timestamp_column_name=column_mapping.get('timestamp', None),
            chunker=chunker,
            metrics=metrics,
            problem_type=problem_type,
        ).fit(reference_data)

        if console:
            console.log('calculating on analysis data')
        results = calc.calculate(analysis_data)

        plots = {}
        if isinstance(writer, RawFilesWriter):
            if console:
                console.log('generating result plots')
            plots = {
                f'realized_{metric}': results.plot(kind, metric=metric)
                for kind in ['performance']
                for metric in metrics
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
    writer.write(result=results, plots=plots, calculator_name='realized_performance')


def _run_cbpe_performance_estimation(
    reference_data: pd.DataFrame,
    analysis_data: pd.DataFrame,
    column_mapping: Dict[str, Any],
    problem_type: ProblemType,
    chunker: Chunker,
    writer: Writer,
    ignore_errors: bool,
    console: Console = None,
):
    if console:
        console.rule('[cyan]Confidence Base Performance Estimator[/]')

    if problem_type not in [ProblemType.CLASSIFICATION_BINARY, ProblemType.CLASSIFICATION_MULTICLASS]:
        _logger.info(f"CBPE does not support '{problem_type.name}' problems. Skipping CBPE estimation.")
        if console:
            console.log(
                f"CBPE does not support '{problem_type.name}' problems. Skipping CBPE estimation.",
                style='yellow',
            )
        return

    metrics = ['roc_auc', 'f1', 'precision', 'recall', 'specificity', 'accuracy']

    try:
        if console:
            console.log('fitting on reference data')
        estimator = CBPE(  # type: ignore
            y_true=column_mapping['y_true'],
            y_pred=column_mapping['y_pred'],
            y_pred_proba=column_mapping['y_pred_proba'],
            timestamp_column_name=column_mapping.get('timestamp', None),
            problem_type=problem_type,
            chunker=chunker,
            metrics=metrics,
        ).fit(reference_data)
        if console:
            console.log('estimating on analysis data')
        results = estimator.estimate(analysis_data)

        plots = {}
        if isinstance(writer, RawFilesWriter):
            if console:
                console.log('generating result plots')
            plots = {
                f'estimated_{metric}': results.plot(kind, metric=metric)
                for kind in ['performance']
                for metric in metrics
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
    writer.write(result=results, plots=plots, calculator_name='confidence_based_performance_estimator')


def _run_dee_performance_estimation(
    reference_data: pd.DataFrame,
    analysis_data: pd.DataFrame,
    column_mapping: Dict[str, Any],
    problem_type: ProblemType,
    chunker: Chunker,
    writer: Writer,
    ignore_errors: bool,
    console: Console = None,
):
    if console:
        console.rule('[cyan]Direct Loss Estimator[/]')

    if problem_type not in [ProblemType.REGRESSION]:
        _logger.info(f"DLE does not support '{problem_type.name}' problems. Skipping DLE estimation.")
        if console:
            console.log(
                f"DLE does not support '{problem_type.name}' problems. Skipping DLE estimation.",
                style='yellow',
            )
        return

    try:
        if console:
            console.log('fitting on reference data')
        estimator = DLE(  # type: ignore
            feature_column_names=column_mapping['features'],
            y_true=column_mapping['y_true'],
            y_pred=column_mapping['y_pred'],
            timestamp_column_name=column_mapping.get('timestamp', None),
            chunker=chunker,
            metrics=DEFAULT_METRICS,
        ).fit(reference_data)
        if console:
            console.log('estimating on analysis data')
        results = estimator.estimate(analysis_data)

        plots = {}
        if isinstance(writer, RawFilesWriter):
            if console:
                console.log('generating result plots')
            plots = {
                f'estimated_{metric}': results.plot(kind, metric=metric)
                for kind in ['performance']
                for metric in DEFAULT_METRICS
            }
    except Exception as exc:
        msg = f"Failed to run DLE performance estimator: {exc}"
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
    writer.write(result=results, plots=plots, calculator_name='direct_error_estimator')
