#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing the performance calculation implementation.

For more information, check out the `tutorials`_.

.. _tutorials:
    https://nannyml.readthedocs.io/en/stable/tutorials/performance_calculation.html
"""

SUPPORTED_CLASSIFICATION_METRIC_VALUES = [
    'roc_auc',
    'f1',
    'precision',
    'recall',
    'specificity',
    'accuracy',
    'confusion_matrix',
    'business_value',
    'average_precision',
]

SUPPORTED_REGRESSION_METRIC_VALUES = [
    'mae',
    'mape',
    'mse',
    'msle',
    'rmse',
    'rmsle',
]

SUPPORTED_METRIC_VALUES = SUPPORTED_CLASSIFICATION_METRIC_VALUES + SUPPORTED_REGRESSION_METRIC_VALUES

SUPPORTED_METRIC_FILTER_VALUES = SUPPORTED_METRIC_VALUES + [
    'true_positive',
    'true_negative',
    'false_positive',
    'false_negative',
]

from .calculator import PerformanceCalculator, Result  # noqa: E402
from .metrics import Metric, MetricFactory  # noqa: E402
