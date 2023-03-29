#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Used as an access point to start using NannyML in its most simple form."""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import jinja2
import pandas as pd
from rich.console import Console

from nannyml._typing import Result
from nannyml.config import CalculatorConfig, Config, InputConfig, InputDataConfig, StoreConfig, WriterConfig
from nannyml.drift.multivariate.data_reconstruction import DataReconstructionDriftCalculator
from nannyml.drift.univariate import UnivariateDriftCalculator
from nannyml.exceptions import InvalidArgumentsException, IOException
from nannyml.io import FileReader, FilesystemStore, Writer, WriterFactory
from nannyml.io.store import Store
from nannyml.performance_calculation import PerformanceCalculator
from nannyml.performance_estimation.confidence_based import CBPE
from nannyml.performance_estimation.direct_loss_estimation import DLE


@dataclass
class CalculatorContext:
    calculator: Dict[str, Any]
    input: Dict[str, Any]
    outputs: Optional[List[Dict[str, Any]]]
    store: Optional[Dict[str, Any]]
    result: Optional[Result]


@contextmanager
def calculator_context(calculator_config: CalculatorConfig, input_config: InputConfig):
    outputs = [_add_rendered_path(c.dict()) for c in (calculator_config.outputs or [])]
    inputs = _add_rendered_path(input_config.dict())
    store = _add_rendered_path(calculator_config.store.dict()) if calculator_config.store else None

    yield CalculatorContext(
        calculator=calculator_config.dict(),
        outputs=outputs,
        input=inputs,
        store=store,
        result=None,
    )


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
    on_fit: Optional[Callable[[CalculatorContext], Any]] = None,
    on_calculate: Optional[Callable[[CalculatorContext], Any]] = None,
    on_success: Optional[Callable[[CalculatorContext], Any]] = None,
    on_fail: Optional[Callable[[CalculatorContext, Optional[Exception]], Any]] = None,
):
    logger = RunnerLogger(logger=logging.getLogger(__name__), console=console)
    try:
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
            with calculator_context(calculator_config, config.input) as context:
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
                        if on_fit:
                            on_fit(context)
                        calc.fit(reference_data)
                        store.store(obj=calc, filename=calculator_config.store.filename)
                else:
                    calc = calc_cls(**calculator_config.params)
                    if on_fit:
                        on_fit(context)
                    calc.fit(reference_data)

                if on_calculate:
                    on_calculate(context)
                result = calc.calculate(analysis_data) if hasattr(calc, 'calculate') else calc.estimate(analysis_data)
                context.result = result

                for writer, write_args in writers:
                    writer.write(result, **write_args)

                if on_success:
                    on_success(context)
    except Exception as exc:
        raise exc


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


def _get_ignore_errors(ignore_errors: bool, config: Config) -> bool:
    if ignore_errors is None:
        if config.ignore_errors is None:
            return False
        else:
            return config.ignore_errors
    else:
        return ignore_errors


def _add_rendered_path(config: Dict[str, Any]) -> Dict[str, Any]:
    if 'params' in config and 'path' in config['params']:
        config['params']['rendered_path'] = _render_path_template(config['params']['path'])

    return config


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
