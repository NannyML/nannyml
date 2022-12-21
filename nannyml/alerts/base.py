#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import logging
import typing
from enum import Enum
from typing import Callable, Dict, List, Union

from nannyml._typing import Result
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

        if key not in cls.registry:
            raise InvalidArgumentsException(f"unknown metric key '{key}' given. " "Should be one of ['slack'].")

        handler_class = cls.registry[key]
        return handler_class(**kwargs)  # type: ignore

    @classmethod
    def register(cls, key: str) -> Callable:
        def inner_wrapper(wrapped_class: AlertHandler) -> AlertHandler:
            if key in cls.registry:
                cls._logger().warning(f"an AlertHandler was already registered for key {key} and will be replaced.")
            cls.registry[key] = wrapped_class
            return wrapped_class

        return inner_wrapper
