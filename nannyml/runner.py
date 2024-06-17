#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Used as an access point to start using NannyML in its most simple form."""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import pandas as pd
from rich.console import Console

from nannyml._typing import Calculator, Estimator, Result
from nannyml.config import Config, InputDataConfig, StoreConfig, WriterConfig
from nannyml.data_quality.missing import MissingValuesCalculator
from nannyml.data_quality.unseen import UnseenValuesCalculator
from nannyml.distribution.categorical import CategoricalDistributionCalculator
from nannyml.distribution.continuous import ContinuousDistributionCalculator
from nannyml.drift.multivariate.data_reconstruction import DataReconstructionDriftCalculator
from nannyml.drift.multivariate.domain_classifier import DomainClassifierCalculator
from nannyml.drift.univariate import UnivariateDriftCalculator
from nannyml.exceptions import InvalidArgumentsException
from nannyml.io import FileReader, FilesystemStore, Writer, WriterFactory
from nannyml.io.store import Store
from nannyml.performance_calculation import PerformanceCalculator
from nannyml.performance_estimation.confidence_based import CBPE
from nannyml.performance_estimation.direct_loss_estimation import DLE
from nannyml.stats.avg.calculator import SummaryStatsAvgCalculator
from nannyml.stats.count.calculator import SummaryStatsRowCountCalculator
from nannyml.stats.median.calculator import SummaryStatsMedianCalculator
from nannyml.stats.std.calculator import SummaryStatsStdCalculator
from nannyml.stats.sum.calculator import SummaryStatsSumCalculator


@dataclass
class RunContext:
    STEPS_PER_CALCULATOR = 3

    current_step: int
    total_steps: int
    current_calculator: str
    current_calculator_config: Optional[Dict[str, Any]] = None
    current_calculator_success: bool = True
    run_success: bool = True
    result: Optional[Result] = None

    def increase_step(self):
        self.current_step += 1


@dataclass
class RunInput:
    reference_data: pd.DataFrame
    analysis_data: pd.DataFrame
    target_data: Optional[pd.DataFrame] = None
    target_join_column: Optional[str] = None


@contextmanager
def run_context(config: Config):
    yield RunContext(
        current_step=0,
        total_steps=len([c for c in config.calculators if c.enabled]) * RunContext.STEPS_PER_CALCULATOR,
        current_calculator='',
    )


_logger = logging.getLogger(__name__)


class CalculatorFactory:
    """A factory class that produces Metric instances based on a given magic string or a metric specification."""

    registry: Dict[str, Type] = {
        'univariate_drift': UnivariateDriftCalculator,
        'reconstruction_error': DataReconstructionDriftCalculator,
        'domain_classifier': DomainClassifierCalculator,
        'performance': PerformanceCalculator,
        'cbpe': CBPE,
        'dle': DLE,
        'missing_values': MissingValuesCalculator,
        'unseen_values': UnseenValuesCalculator,
        'continuous_distribution': ContinuousDistributionCalculator,
        'categorical_distribution': CategoricalDistributionCalculator,
        'summary_stats_avg': SummaryStatsAvgCalculator,
        'summary_stats_row_count': SummaryStatsRowCountCalculator,
        'summary_stats_median': SummaryStatsMedianCalculator,
        'summary_stats_std': SummaryStatsStdCalculator,
        'summary_stats_sum': SummaryStatsSumCalculator,
    }

    @classmethod
    def register(cls, name: str, calculator_type: Union[Type[Calculator], Type[Estimator]]):
        cls.registry[name] = calculator_type


class RunnerLogger:
    def __init__(self, logger: logging.Logger, console: Optional[Console] = None):
        self.logger = logger
        self.console = console

    def log(self, message: object, log_level: int = logging.INFO):
        if self.logger:
            self.logger.log(level=log_level, msg=message)

        if self.console and log_level == logging.INFO:
            self.console.log(message)


def run(  # noqa: C901
    config: Config,
    input: Optional[RunInput] = None,
    logger: logging.Logger = logging.getLogger(__name__),
    console: Optional[Console] = None,
    on_fit: Optional[Callable[[RunContext], Any]] = None,
    on_calculate: Optional[Callable[[RunContext], Any]] = None,
    on_write: Optional[Callable[[RunContext], Any]] = None,
    on_calculator_complete: Optional[Callable[[RunContext], Any]] = None,
    on_run_complete: Optional[Callable[[RunContext], Any]] = None,
    on_fail: Optional[Callable[[RunContext, Optional[Exception]], Any]] = None,
):
    run_logger = RunnerLogger(logger, console)
    try:
        with run_context(config) as context:
            if input is not None:
                if config.input is not None:
                    raise InvalidArgumentsException("Both config.input and input provided. Please provide only one")

                reference_data = input.reference_data
                analysis_data = input.analysis_data
                if input.target_data is not None:
                    analysis_data = _add_targets_to_analysis_data(
                        analysis_data, input.target_data, input.target_join_column
                    )
            elif config.input is not None:
                run_logger.log("reading reference data", log_level=logging.DEBUG)
                reference_data = read_data(config.input.reference_data, run_logger)

                # read analysis data
                run_logger.log("reading analysis data", log_level=logging.DEBUG)
                analysis_data = read_data(config.input.analysis_data, run_logger)

                if config.input.target_data:
                    run_logger.log("reading target data", log_level=logging.DEBUG)
                    target_data = read_data(config.input.target_data, run_logger)
                    analysis_data = _add_targets_to_analysis_data(
                        analysis_data, target_data, config.input.target_data.join_column
                    )
            else:
                raise InvalidArgumentsException("no input data provided")

            for calculator_config in config.calculators:
                try:
                    context.current_calculator_config = calculator_config.dict()
                    context.current_calculator = calculator_config.name or calculator_config.type

                    if not calculator_config.enabled:
                        continue

                    writers = get_output_writers(calculator_config.outputs, run_logger)

                    store = get_store(calculator_config.store, run_logger)

                    if calculator_config.type not in CalculatorFactory.registry:
                        raise InvalidArgumentsException(f"unknown calculator type '{calculator_config.type}'")

                    # first step: load or (create + fit) calculator
                    context.increase_step()
                    calc_cls = CalculatorFactory.registry[calculator_config.type]
                    if store and calculator_config.store:
                        run_logger.log(
                            f"[{context.current_step}/{context.total_steps}] '{context.current_calculator}': "
                            f"loading calculator from store"
                        )
                        if calculator_config.store.invalidate:
                            calc = None
                        else:
                            calc = store.load(filename=calculator_config.store.filename, as_type=calc_cls)

                        if calc is None:
                            reason = 'invalidated' if calculator_config.store.invalidate else 'not found in store'
                            run_logger.log(
                                f"calculator '{context.current_calculator}' {reason}. "
                                f"Creating, fitting and storing new instance",
                                log_level=logging.DEBUG,
                            )
                            calc = calc_cls(**calculator_config.params)
                            if on_fit:
                                on_fit(context)
                            calc.fit(reference_data)
                            store.store(obj=calc, filename=calculator_config.store.filename)
                    else:
                        run_logger.log(
                            f"[{context.current_step}/{context.total_steps}] '{context.current_calculator}': "
                            f"creating and fitting calculator"
                        )
                        calc = calc_cls(**calculator_config.params)
                        if on_fit:
                            on_fit(context)
                        calc.fit(reference_data)

                    # second step: perform calculations
                    context.increase_step()
                    run_logger.log(
                        f"[{context.current_step}/{context.total_steps}] '{context.current_calculator}': "
                        f"running calculator"
                    )
                    if on_calculate:
                        on_calculate(context)
                    result = (
                        calc.calculate(analysis_data) if hasattr(calc, 'calculate') else calc.estimate(analysis_data)
                    )
                    context.result = result

                    # third step: write results
                    context.increase_step()
                    run_logger.log(
                        f"[{context.current_step}/{context.total_steps}] '{context.current_calculator}': "
                        f"writing out results"
                    )
                    if on_write:
                        on_write(context)

                    for writer, write_args in writers:
                        run_logger.log(f"writing results with {writer} and args {write_args}", log_level=logging.DEBUG)
                        writer.write(result, **write_args)

                    if on_calculator_complete:
                        on_calculator_complete(context)
                except Exception as exc:
                    context.current_calculator_success = False
                    context.run_success = False
                    if on_fail:
                        on_fail(context, exc)
                    run_logger.log(f"an unexpected exception occurred running '{calculator_config.type}': {exc}")

            if on_run_complete:
                on_run_complete(context)

    except Exception as exc:
        context.current_calculator = None
        context.current_calculator_config = None
        context.run_success = False
        if on_fail:
            on_fail(context, exc)

        raise exc


def read_data(input_config: InputDataConfig, logger: Optional[RunnerLogger] = None) -> pd.DataFrame:
    data = FileReader(
        filepath=input_config.path, credentials=input_config.credentials, read_args=input_config.read_args
    ).read()
    if logger:
        logger.log(f"read {data.size} rows from {input_config.path}")
    return data


def get_output_writers(
    outputs_config: Optional[List[WriterConfig]], logger: Optional[RunnerLogger] = None
) -> List[Tuple[Writer, Dict[str, Any]]]:
    if not outputs_config:
        return []

    writers: List[Tuple[Writer, Dict[str, Any]]] = []

    for writer_config in outputs_config:
        writer = WriterFactory.create(writer_config.type, writer_config.params)
        writers.append((writer, writer_config.write_args or {}))

    return writers


def get_store(store_config: Optional[StoreConfig], logger: Optional[RunnerLogger] = None) -> Optional[Store]:
    if store_config:
        if logger:
            logger.log(f"using file system store with path '{store_config.path}'", log_level=logging.DEBUG)
        store = FilesystemStore(
            root_path=store_config.path,
            credentials=store_config.credentials or {},
        )
        return store

    return None


def _get_ignore_errors(ignore_errors: bool, config: Config) -> bool:
    if ignore_errors is None:
        if config.ignore_errors is None:
            return False
        else:
            return config.ignore_errors
    else:
        return ignore_errors


def _add_targets_to_analysis_data(
    analysis_data: pd.DataFrame, target_data: pd.DataFrame, join_column: Optional[str]
) -> pd.DataFrame:
    if join_column is not None:
        return analysis_data.merge(target_data, on=join_column)
    else:
        return analysis_data.join(target_data)
