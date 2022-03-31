#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit tests for performance metrics."""
import pytest

from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_calculation.metrics import AUROC, Metric, MetricFactory


@pytest.mark.parametrize('key,metric', [('roc_auc', AUROC())])
def test_metric_factory_returns_correct_metric_given_str_key(key, metric):  # noqa: D103
    sut = MetricFactory.create(key)
    assert sut == metric


def test_metric_factory_raises_invalid_args_exception_when_str_key_unknown():  # noqa: D103
    with pytest.raises(InvalidArgumentsException):
        _ = MetricFactory.create('foo')


def test_metric_factory_returns_metric_when_provided_metric_key():  # noqa: D103
    custom_metric = Metric(
        display_name='custom_metric', column_name='custom_metric', lower_threshold=0.5, upper_threshold=0.9
    )
    sut = MetricFactory.create(custom_metric)
    assert sut.display_name == 'custom_metric'
    assert sut.lower_threshold == 0.5
    assert sut.upper_threshold == 0.9


def test_metric_factory_raises_invalid_args_exception_when_invalid_key_type_given():  # noqa: D103
    with pytest.raises(InvalidArgumentsException):
        _ = MetricFactory.create(123)
