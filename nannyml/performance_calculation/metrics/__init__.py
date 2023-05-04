#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Package containing the implementation of realized performance metrics.

The performance calculator manages a list of :class:`~nannyml.performance_calculation.metrics.base.Metric` instances,
constructed using the :class:`~nannyml.performance_calculation.metrics.base.MetricFactory`.
The estimator is then responsible for delegating the `fit` and `estimate` method calls to each of the managed
:class:`~nannyml.performance_calculation.metrics.base.Metric` instances and building a
:class:`~nannyml.performance_calculation.result.Result` object.

For more information, check out the `tutorials`_.

.. _tutorials:
    https://nannyml.readthedocs.io/en/stable/tutorials/performance_calculation.html
"""

from .base import Metric, MetricFactory
from .binary_classification import (
    BinaryClassificationAccuracy,
    BinaryClassificationAUROC,
    BinaryClassificationF1,
    BinaryClassificationPrecision,
    BinaryClassificationRecall,
    BinaryClassificationSpecificity,
)
from .multiclass_classification import (
    MulticlassClassificationAccuracy,
    MulticlassClassificationAUROC,
    MulticlassClassificationF1,
    MulticlassClassificationPrecision,
    MulticlassClassificationRecall,
    MulticlassClassificationSpecificity,
)
from .regression import MAE
