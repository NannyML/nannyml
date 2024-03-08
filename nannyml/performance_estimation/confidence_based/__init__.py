#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Package containing the Confidence Based Performance Estimator (CBPE).

Assuming properly calibrated probabilities, confusion matrix elements can be estimated and then used to
calculate any performance metric. Given the assumptions are met, CBPE provides an unbiased estimation of the
performance of the monitored model based on the monitored model’s outputs only (i.e. without access to targets).

For more information, check out the `tutorial for binary classification`_,
the `tutorial for multiclass classification`_ or the `deep dive`_.

.. _tutorial for binary classification:
    https://nannyml.readthedocs.io/en/stable/tutorials/performance_estimation/binary_performance_estimation.html

.. _tutorial for multiclass classification:
    https://nannyml.readthedocs.io/en/stable/tutorials/performance_estimation/multiclass_performance_estimation.html

.. _deep dive:
    https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html#confidence-based-performance-estimation-cbpe
"""

SUPPORTED_METRIC_VALUES = [
    'roc_auc',
    'f1',
    'average_precision',
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
