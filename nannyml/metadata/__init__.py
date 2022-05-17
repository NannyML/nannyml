#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""The package containing all functionality to define, extract and manipulate model metadata."""

from .base import ModelMetadata, ModelType
from .binary_classification import BinaryClassificationMetadata
from .extraction import extract_metadata
from .feature import Feature, FeatureType
from .multiclass_classification import MulticlassClassificationMetadata
from .regression import RegressionMetadata
