#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Package containing the Confidence Based Performance Estimator (CBPE)."""

SUPPORTED_METRIC_VALUES = [
    'roc_auc',
    'f1',
    'precision',
    'recall',
    'specificity',
    'accuracy',
    'confusion_matrix',
    'business_value',
]

SUPPORTED_METRIC_FILTER_VALUES = SUPPORTED_METRIC_VALUES + [
    'true_positive',
    'true_negative',
    'false_positive',
    'false_negative',
]


from .cbpe import CBPE  # noqa: E402
from .results import Result  # noqa: E402
