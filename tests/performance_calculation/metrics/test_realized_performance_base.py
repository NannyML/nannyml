#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import pytest

from nannyml._typing import UseCase
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_calculation.metrics import MetricFactory


@pytest.mark.parametrize('use_case', [UseCase.CLASSIFICATION_BINARY, UseCase.CLASSIFICATION_MULTICLASS])
def test_metric_factory_raises_invalid_args_exception_when_key_unknown(use_case):  # noqa: D103
    with pytest.raises(InvalidArgumentsException):
        _ = MetricFactory.create('foo', use_case)


@pytest.mark.parametrize('use_case', [UseCase.CLASSIFICATION_BINARY, UseCase.CLASSIFICATION_MULTICLASS])
def test_metric_factory_raises_invalid_args_exception_when_invalid_key_type_given(use_case):  # noqa: D103
    with pytest.raises(InvalidArgumentsException):
        _ = MetricFactory.create(123, use_case)
