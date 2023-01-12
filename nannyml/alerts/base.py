#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import logging
import typing
import warnings
from enum import Enum
from typing import Callable, Dict, List, Union

from nannyml._typing import Result
from nannyml.drift.univariate import Result as UnivariateDriftResult
from nannyml.exceptions import InvalidArgumentsException

if typing.TYPE_CHECKING:
    from typing_extensions import Protocol
else:
    Protocol = object


class AlertType(str, Enum):
    PERFORMANCE = 'performance'
    DRIFT = 'drift'


class AlertHandler(Protocol):
    def handle(self, results: Union[Result, List[Result]], only_alerts: bool = True):
        ...


class AlertHandlerFactory:
    """A factory class that produces Metric instances based on a given magic string or a metric specification."""

    registry: Dict[str, AlertHandler] = {}

    @classmethod
    def _logger(cls) -> logging.Logger:
        return logging.getLogger(__name__)

    @classmethod
    def create(cls, key: str, **kwargs) -> AlertHandler:
        """Returns a Metric instance for a given key."""
        if not isinstance(key, str):
            raise InvalidArgumentsException(f"cannot create handler given a '{type(key)}'" "Please provide a string")

        available_keys = ', '.join(map(lambda k: f"'{k}'", cls.registry.keys()))
        if key not in cls.registry:
            raise InvalidArgumentsException(
                f"unknown metric key '{key}' given. " f"Should be one of [{available_keys}]."
            )

        handler_class = cls.registry[key]
        return handler_class(**kwargs)  # type: ignore

    @classmethod
    def register(cls, key: str) -> Callable:
        def inner_wrapper(wrapped_class: AlertHandler) -> AlertHandler:
            if key in cls.registry:
                msg = f"an AlertHandler was already registered for key {key} and will be replaced."
                warnings.warn(msg, UserWarning)
                cls._logger().warning(msg)
            cls.registry[key] = wrapped_class
            return wrapped_class

        return inner_wrapper


def get_column_names_with_alerts(result: UnivariateDriftResult) -> List[str]:
    df = result.filter(period='analysis').to_df()

    columns_with_alerts = set()
    for column_name in result.column_names:
        alert_cols = df.loc[:, (column_name, slice(None), 'alert')].columns
        for col in alert_cols:
            has_alerts = df.get(col).any()
            if has_alerts:
                columns_with_alerts.add(column_name)

    return list(columns_with_alerts)


def get_metrics_with_alerts(result: Result) -> List[str]:
    df = result.filter(period='analysis').to_df()

    metrics_with_alerts = set()
    if hasattr(result, 'metrics'):
        for metric in result.metrics:
            has_alerts = df.get((metric.column_name, 'alert')).any()
            if has_alerts:
                metrics_with_alerts.add(metric.display_name)

    return list(metrics_with_alerts)
