#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

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
