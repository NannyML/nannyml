#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Used as an access point to start using NannyML in its most simple form."""

import logging
import sys
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import jinja2
import pandas as pd
from rich.console import Console

from nannyml._typing import ProblemType
from nannyml.chunk import Chunker
from nannyml.config import Config, InputDataConfig, StoreConfig, WriterConfig
from nannyml.drift.multivariate.data_reconstruction import DataReconstructionDriftCalculator
from nannyml.drift.univariate import UnivariateDriftCalculator
from nannyml.exceptions import InvalidArgumentsException, IOException
from nannyml.io import FileReader, FilesystemStore, RawFilesWriter, Writer, WriterFactory
from nannyml.io.store import Store
from nannyml.performance_calculation import (
    SUPPORTED_CLASSIFICATION_METRIC_VALUES,
    SUPPORTED_REGRESSION_METRIC_VALUES,
    PerformanceCalculator,
)
from nannyml.performance_estimation.confidence_based import CBPE
from nannyml.performance_estimation.confidence_based import SUPPORTED_METRIC_VALUES as CBPE_SUPPORTED_METRICS
from nannyml.performance_estimation.direct_loss_estimation import DLE
from nannyml.performance_estimation.direct_loss_estimation import SUPPORTED_METRIC_VALUES as DLE_SUPPORTED_METRICS


class Runner:
    pass


_registry: Dict[str, Type] = {
    'univariate_drift': UnivariateDriftCalculator,
    'multivariate_drift': DataReconstructionDriftCalculator,
    'performance': PerformanceCalculator,
    'cbpe': CBPE,
    'dle': DLE,
}
_logger = logging.getLogger(__name__)


class RunnerLogger:
    def __init__(
        self, logger: Optional[logging.Logger] = logging.getLogger(__name__), console: Optional[Console] = None
    ):
        self.logger = logger
        self.console = console

    def log(self, message: object, log_level: int = logging.INFO):
        if self.logger:
            self.logger.log(level=log_level, msg=message)

        if self.console:
            self.console.log(message)


def run(
    config: Config,
    console: Optional[Console] = None,
    on_fit: Optional[Callable] = None,
    on_calculate: Optional[Callable] = None,
    on_success: Optional[Callable] = None,
    on_fail: Optional[Callable] = None,
):
    logger = RunnerLogger(logger=logging.getLogger(__name__), console=console)

    logger.log("reading reference data")
    reference_data = read_data(config.input.reference_data, logger)

    # read analysis data
    logger.log("reading analysis data")
    analysis_data = read_data(config.input.analysis_data, logger)

    if config.input.target_data:
        logger.log("reading target data")
        target_data = read_data(config.input.target_data, logger)

        if config.input.target_data.join_column:
            analysis_data = analysis_data.merge(target_data, on=config.input.target_data.join_column)
        else:
            analysis_data = analysis_data.join(target_data)

    for calculator_config in config.calculators:
        if not calculator_config.enabled:
            continue

        writers = get_output_writers(calculator_config.outputs, logger)

        store = get_store(calculator_config.store, logger)

        if calculator_config.type not in _registry:
            raise InvalidArgumentsException(f"unknown calculator type '{calculator_config.type}'")

        calc_cls = _registry[calculator_config.type]
        if store and calculator_config.store:
            calc = store.load(filename=calculator_config.store.filename, as_type=calc_cls)
            if calc is None:
                calc = calc_cls(**calculator_config.params)
                calc.fit(reference_data)
                store.store(obj=calc, filename=calculator_config.store.filename)
        else:
            calc = calc_cls(**calculator_config.params)
            calc.fit(reference_data)

        result = calc.calculate(analysis_data) if hasattr(calc, 'calculate') else calc.estimate(analysis_data)

        for writer, write_args in writers:
            writer.write(result, **write_args)


def read_data(input_config: InputDataConfig, logger: Optional[RunnerLogger] = None) -> pd.DataFrame:
    path = _render_path_template(input_config.path)
    data = FileReader(filepath=path, credentials=input_config.credentials, read_args=input_config.read_args).read()
    if logger:
        logger.log(f"read {data.size} rows from {path}")
    return data


def get_output_writers(
    outputs_config: Optional[List[WriterConfig]], logger: Optional[RunnerLogger] = None
) -> List[Tuple[Writer, Dict[str, Any]]]:
    if not outputs_config:
        return []

    writers: List[Tuple[Writer, Dict[str, Any]]] = []

    for writer_config in outputs_config:
        if writer_config.params and 'path' in writer_config.params:
            writer_config.params['path'] = _render_path_template(writer_config.params['path'])

        writer = WriterFactory.create(writer_config.type, writer_config.params)
        writers.append((writer, writer_config.write_args or {}))

    return writers


def get_store(store_config: Optional[StoreConfig], logger: Optional[RunnerLogger] = None) -> Optional[Store]:
    if store_config:
        path = _render_path_template(store_config.path)
        if logger:
            logger.log(f"using file system store with path '{path}'")
        store = FilesystemStore(
            root_path=path,
            credentials=store_config.credentials or {},
        )
        return store

    return None


def _run_statistical_univariate_feature_drift_calculator(
    reference_data: pd.DataFrame,
    analysis_data: pd.DataFrame,
    column_mapping: Dict[str, Any],
    problem_type: ProblemType,
    chunker: Chunker,
    writer: Writer,
    store: Optional[Store],
    ignore_errors: bool,
    console: Optional[Console] = None,
):
    if console:
        console.rule('[cyan]UnivariateStatisticalDriftCalculator[/]')
    try:
        calc: Optional[UnivariateDriftCalculator] = None  # calculator to load or create
        calc_path = 'univariate_drift/calculator.pkl'  # the path to load or store the calculator in the store

        if store:  # we have a store defined, let's try to load the fitted calculator from there
            if console:
                console.log('loading calculator from store')
            calc = store.load(path=calc_path, as_type=UnivariateDriftCalculator)

        if not calc:  # no store or no fitted calculator was in the store
            if console:
                console.log('no fitted calculator found in store')
                console.log('fitting new calculator on reference data')
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
                categorical_methods=['chi2', 'jensen_shannon', 'l_infinity'],
                continuous_methods=['kolmogorov_smirnov', 'jensen_shannon', 'wasserstein'],
            )
            calc.fit(reference_data)

            if store:
                store.store(calc, path=calc_path)
                if console:
                    console.log('storing fitted calculator to store')

        # raise RuntimeError("🔥 something's not right there... 🔥")

        if console:
            console.log('calculating on analysis data')
        results = calc.calculate(analysis_data)

        plots = {}
        if isinstance(writer, RawFilesWriter):
            if console:
                console.log('generating result plots')
            plots = {f'{kind}': results.plot(kind=kind) for kind in ['drift', 'distribution']}
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
    store: Optional[Store],
    ignore_errors: bool,
    console: Optional[Console] = None,
):
    if console:
        console.rule('[cyan]DataReconstructionDriftCalculator[/]')
    try:
        calc: Optional[DataReconstructionDriftCalculator] = None  # calculator to load or create
        calc_path = 'data_reconstruction/calculator.pkl'  # the path to load or store the calculator in the store

        if store:  # we have a store defined, let's try to load the fitted calculator from there
            if console:
                console.log('loading calculator from store')
            calc = store.load(path=calc_path, as_type=DataReconstructionDriftCalculator)

        if not calc:  # no store or no fitted calculator was in the store
            if console:
                console.log('no fitted calculator found in store')
                console.log('fitting new calculator on reference data')
            calc = DataReconstructionDriftCalculator(
                column_names=column_mapping['features'],
                timestamp_column_name=column_mapping.get('timestamp', None),
                chunker=chunker,
            )
            calc.fit(reference_data)
            if store:
                store.store(calc, path=calc_path)
                if console:
                    console.log('storing fitted calculator to store')

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


def _run_realized_performance_calculator(  # noqa: C901
    reference_data: pd.DataFrame,
    analysis_data: pd.DataFrame,
    column_mapping: Dict[str, Any],
    problem_type: ProblemType,
    chunker: Chunker,
    writer: Writer,
    store: Optional[Store],
    ignore_errors: bool,
    console: Optional[Console] = None,
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
    try:
        calc: Optional[PerformanceCalculator] = None  # calculator to load or create
        calc_path = 'realized_performance/calculator.pkl'  # the path to load or store the calculator in the store

        if store:  # we have a store defined, let's try to load the fitted calculator from there
            if console:
                console.log('loading calculator from store')
            calc = store.load(path=calc_path, as_type=PerformanceCalculator)

        if not calc:  # no store or no fitted calculator was in the store
            if problem_type in [ProblemType.CLASSIFICATION_BINARY]:
                # requires a non-default parameter 'business_value_matrix'
                metrics = [
                    metric for metric in SUPPORTED_CLASSIFICATION_METRIC_VALUES if metric not in ['business_value']
                ]
            elif problem_type in [ProblemType.CLASSIFICATION_MULTICLASS]:
                metrics = [
                    metric
                    for metric in SUPPORTED_CLASSIFICATION_METRIC_VALUES
                    if metric not in ['business_value', 'confusion_matrix']
                ]
            elif problem_type in [ProblemType.REGRESSION]:
                metrics = SUPPORTED_REGRESSION_METRIC_VALUES
            else:
                raise InvalidArgumentsException(f"unsupported problem type '{problem_type}'")

            if console:
                console.log('no fitted calculator found in store')
                console.log('fitting new calculator on reference data')
            calc = PerformanceCalculator(
                y_true=column_mapping['y_true'],
                y_pred=column_mapping['y_pred'],
                y_pred_proba=column_mapping.get('y_pred_proba', None),
                timestamp_column_name=column_mapping.get('timestamp', None),
                chunker=chunker,
                metrics=metrics,
                problem_type=problem_type,
            )
            calc.fit(reference_data)
            if store:
                store.store(calc, path=calc_path)
                if console:
                    console.log('storing fitted calculator to store')

        if console:
            console.log('calculating on analysis data')
        results = calc.calculate(analysis_data)

        plots = {}
        if isinstance(writer, RawFilesWriter):
            if console:
                console.log('generating result plots')
            plots = {f'{kind}': results.plot(kind) for kind in ['performance']}
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


def _run_cbpe_performance_estimation(  # noqa: C901
    reference_data: pd.DataFrame,
    analysis_data: pd.DataFrame,
    column_mapping: Dict[str, Any],
    problem_type: ProblemType,
    chunker: Chunker,
    writer: Writer,
    store: Optional[Store],
    ignore_errors: bool,
    console: Optional[Console] = None,
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

    try:
        estimator: Optional[CBPE] = None  # estimator to load or create
        estimator_path = 'cbpe/estimator.pkl'  # the path to load or store the estimator in the store

        if store:  # we have a store defined, let's try to load the fitted calculator from there
            if console:
                console.log('loading estimator from store')
            estimator = store.load(path=estimator_path, as_type=CBPE)

        if not estimator:  # no store or no fitted calculator was in the store
            if problem_type in [ProblemType.CLASSIFICATION_BINARY]:
                # requires a non-default parameter 'business_value_matrix'
                metrics = [metric for metric in CBPE_SUPPORTED_METRICS if metric not in ['business_value']]
            elif problem_type in [ProblemType.CLASSIFICATION_MULTICLASS]:
                metrics = [
                    metric for metric in CBPE_SUPPORTED_METRICS if metric not in ['business_value', 'confusion_matrix']
                ]
            else:
                raise InvalidArgumentsException(f"unsupported problem type '{problem_type}'")

            if console:
                console.log('no fitted estimator found in store')
                console.log('fitting new estimator on reference data')
            estimator = CBPE(
                y_true=column_mapping['y_true'],
                y_pred=column_mapping['y_pred'],
                y_pred_proba=column_mapping['y_pred_proba'],
                timestamp_column_name=column_mapping.get('timestamp', None),
                problem_type=problem_type,
                chunker=chunker,
                metrics=metrics,
            )
            estimator.fit(reference_data)
            if store:
                store.store(estimator, path=estimator_path)
                if console:
                    console.log('storing fitted estimator to store')

        if console:
            console.log('estimating on analysis data')
        results = estimator.estimate(analysis_data)

        plots = {}
        if isinstance(writer, RawFilesWriter):
            if console:
                console.log('generating result plots')
            plots = {f'{kind}': results.plot(kind) for kind in ['performance']}

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


def _run_dle_performance_estimation(
    reference_data: pd.DataFrame,
    analysis_data: pd.DataFrame,
    column_mapping: Dict[str, Any],
    problem_type: ProblemType,
    chunker: Chunker,
    writer: Writer,
    store: Optional[Store],
    ignore_errors: bool,
    console: Optional[Console] = None,
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
        estimator: Optional[DLE] = None  # estimator to load or create
        estimator_path = 'cbpe/estimator.pkl'  # the path to load or store the estimator in the store

        if store:  # we have a store defined, let's try to load the fitted calculator from there
            if console:
                console.log('loading estimator from store')
            estimator = store.load(path=estimator_path, as_type=DLE)

        if not estimator:  # no store or no fitted calculator was in the store
            if console:
                console.log('no fitted estimator found in store')
                console.log('fitting new estimator on reference data')
            estimator = DLE(
                feature_column_names=column_mapping['features'],
                y_true=column_mapping['y_true'],
                y_pred=column_mapping['y_pred'],
                timestamp_column_name=column_mapping.get('timestamp', None),
                chunker=chunker,
                metrics=DLE_SUPPORTED_METRICS,
            )
            estimator.fit(reference_data)
            if store:
                store.store(estimator, path=estimator_path)
                if console:
                    console.log('storing fitted estimator to store')

        if console:
            console.log('estimating on analysis data')
        results = estimator.estimate(analysis_data)

        plots = {}
        if isinstance(writer, RawFilesWriter):
            if console:
                console.log('generating result plots')
            plots = {f'{kind}': results.plot(kind) for kind in ['performance']}
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
    writer.write(result=results, plots=plots, calculator_name='direct_loss_estimator')


def _get_ignore_errors(ignore_errors: bool, config: Config) -> bool:
    if ignore_errors is None:
        if config.ignore_errors is None:
            return False
        else:
            return config.ignore_errors
    else:
        return ignore_errors


def _render_path_template(path_template: str) -> str:
    try:
        env = jinja2.Environment()
        tpl = env.from_string(path_template)
        return tpl.render(
            minute=datetime.strftime(datetime.today(), "%M"),
            hour=datetime.strftime(datetime.today(), "%H"),
            day=datetime.strftime(datetime.today(), "%d"),
            weeknumber=date.today().isocalendar()[1],
            month=datetime.strftime(datetime.today(), "%m"),
            year=datetime.strftime(datetime.today(), "%Y"),
        )
    except Exception as exc:
        raise IOException(f"could not render file path template: '{path_template}': {exc}")
