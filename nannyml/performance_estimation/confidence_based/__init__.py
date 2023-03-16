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
    'true_positive',
    'true_negative',
    'false_positive',
    'false_negative',
    'true_positive_cost',
    'true_negative_cost',
    'false_positive_cost',
    'false_negative_cost',
    'total_cost',
    'business_cost',
]


from .cbpe import CBPE  # noqa: E402
from .results import Result  # noqa: E402
