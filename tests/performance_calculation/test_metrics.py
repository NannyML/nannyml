#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit tests for performance metrics."""
from typing import Any, Dict, List, Tuple

import pandas as pd
import pytest

from nannyml.exceptions import InvalidArgumentsException
from nannyml.metadata import BinaryClassificationMetadata, ModelMetadata, ModelType, MulticlassClassificationMetadata
from nannyml.performance_calculation import BinaryClassificationAUROC
from nannyml.performance_calculation.metrics import (
    BinaryClassificationAccuracy,
    BinaryClassificationF1,
    BinaryClassificationPrecision,
    BinaryClassificationRecall,
    BinaryClassificationSpecificity,
    MetricFactory,
    MulticlassClassificationAccuracy,
    MulticlassClassificationAUROC,
    MulticlassClassificationF1,
    MulticlassClassificationPrecision,
    MulticlassClassificationRecall,
    MulticlassClassificationSpecificity,
)

binary_classification_metadata = BinaryClassificationMetadata()
multiclass_classification_metadata = MulticlassClassificationMetadata()


@pytest.mark.parametrize(
    'key,metadata,metric',
    [
        ('roc_auc', binary_classification_metadata, BinaryClassificationAUROC(binary_classification_metadata)),
        (
            'roc_auc',
            multiclass_classification_metadata,
            MulticlassClassificationAUROC(multiclass_classification_metadata),
        ),
        ('f1', binary_classification_metadata, BinaryClassificationF1(binary_classification_metadata)),
        ('f1', multiclass_classification_metadata, MulticlassClassificationF1(multiclass_classification_metadata)),
        ('precision', binary_classification_metadata, BinaryClassificationPrecision(binary_classification_metadata)),
        (
            'precision',
            multiclass_classification_metadata,
            MulticlassClassificationPrecision(multiclass_classification_metadata),
        ),
        ('recall', binary_classification_metadata, BinaryClassificationRecall(binary_classification_metadata)),
        (
            'recall',
            multiclass_classification_metadata,
            MulticlassClassificationRecall(multiclass_classification_metadata),
        ),
        (
            'specificity',
            binary_classification_metadata,
            BinaryClassificationSpecificity(binary_classification_metadata),
        ),
        (
            'specificity',
            multiclass_classification_metadata,
            MulticlassClassificationSpecificity(multiclass_classification_metadata),
        ),
        ('accuracy', binary_classification_metadata, BinaryClassificationAccuracy(binary_classification_metadata)),
        (
            'accuracy',
            multiclass_classification_metadata,
            MulticlassClassificationAccuracy(multiclass_classification_metadata),
        ),
    ],
)
def test_metric_factory_returns_correct_metric_given_key_and_metadata(key, metadata, metric):  # noqa: D103
    sut = MetricFactory.create(key, metadata)
    assert sut == metric


@pytest.mark.parametrize('metadata', [BinaryClassificationMetadata(), MulticlassClassificationMetadata()])
def test_metric_factory_raises_invalid_args_exception_when_key_unknown(metadata):  # noqa: D103
    with pytest.raises(InvalidArgumentsException):
        _ = MetricFactory.create('foo', metadata)


@pytest.mark.parametrize('metadata', [BinaryClassificationMetadata(), MulticlassClassificationMetadata()])
def test_metric_factory_raises_invalid_args_exception_when_invalid_key_type_given(metadata):  # noqa: D103
    with pytest.raises(InvalidArgumentsException):
        _ = MetricFactory.create(123, metadata)


def test_metric_factory_raises_runtime_error_when_invalid_metadata_type_given():  # noqa: D103
    class FakeModelMetadata(ModelMetadata):
        def __init__(self):
            super().__init__(model_type=ModelType.CLASSIFICATION_BINARY)

        @property
        def metadata_columns(self):
            return []

        def to_dict(self) -> Dict[str, Any]:
            pass

        def to_df(self) -> pd.DataFrame:
            pass

        def enrich(self, data: pd.DataFrame) -> pd.DataFrame:
            pass

        def is_complete(self) -> Tuple[bool, List[str]]:
            pass

        def extract(self, data: pd.DataFrame, model_name: str = None, exclude_columns: List[str] = None):
            pass

    with pytest.raises(
        RuntimeError,
        match="metric 'roc_auc' is currently not supported for model type "
        "FakeModelMetadata. Please specify another metric or use one of "
        "these supported model types for this metric",
    ):
        _ = MetricFactory.create('roc_auc', FakeModelMetadata())
