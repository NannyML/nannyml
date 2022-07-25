#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit tests for performance metrics."""
from typing import Tuple

import pandas as pd
import pytest

from nannyml import PerformanceCalculator
from nannyml._typing import UseCase
from nannyml.datasets import (
    load_synthetic_binary_classification_dataset,
    load_synthetic_multiclass_classification_dataset,
)
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_calculation.metrics import (
    BinaryClassificationAccuracy,
    BinaryClassificationAUROC,
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


@pytest.fixture
def binary_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, tgt_df = load_synthetic_binary_classification_dataset()
    ref_df['y_pred'] = ref_df['y_pred_proba'].map(lambda p: p >= 0.8).astype(int)
    ana_df['y_pred'] = ana_df['y_pred_proba'].map(lambda p: p >= 0.8).astype(int)

    return ref_df, ana_df, tgt_df


@pytest.fixture()
def performance_calculator() -> PerformanceCalculator:
    return PerformanceCalculator(
        timestamp_column_name='timestamp',
        y_pred_proba='y_pred_proba',
        y_pred='y_pred',
        y_true='y_true',
        metrics=['roc_auc', 'f1'],
    )


@pytest.fixture
def multiclass_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, tgt_df = load_synthetic_multiclass_classification_dataset()

    return ref_df, ana_df, tgt_df


@pytest.mark.parametrize(
    'key,metadata,metric',
    [
        ('roc_auc', UseCase.CLASSIFICATION_BINARY, BinaryClassificationAUROC),
        ('roc_auc', UseCase.CLASSIFICATION_MULTICLASS, MulticlassClassificationAUROC),
        ('f1', UseCase.CLASSIFICATION_BINARY, BinaryClassificationF1),
        ('f1', UseCase.CLASSIFICATION_MULTICLASS, MulticlassClassificationF1),
        ('precision', UseCase.CLASSIFICATION_BINARY, BinaryClassificationPrecision),
        ('precision', UseCase.CLASSIFICATION_MULTICLASS, MulticlassClassificationPrecision),
        ('recall', UseCase.CLASSIFICATION_BINARY, BinaryClassificationRecall),
        ('recall', UseCase.CLASSIFICATION_MULTICLASS, MulticlassClassificationRecall),
        ('specificity', UseCase.CLASSIFICATION_BINARY, BinaryClassificationSpecificity),
        ('specificity', UseCase.CLASSIFICATION_MULTICLASS, MulticlassClassificationSpecificity),
        ('accuracy', UseCase.CLASSIFICATION_BINARY, BinaryClassificationAccuracy),
        ('accuracy', UseCase.CLASSIFICATION_MULTICLASS, MulticlassClassificationAccuracy),
    ],
)
def test_metric_factory_returns_correct_metric_given_key_and_metadata(key, metadata, metric):  # noqa: D103
    calc = PerformanceCalculator(
        timestamp_column_name='timestamp',
        y_pred_proba='y_pred_proba',
        y_pred='y_pred',
        y_true='y_true',
        metrics=['roc_auc', 'f1'],
    )
    sut = MetricFactory.create(key, metadata, {'calculator': calc})
    assert sut == metric(calculator=calc)


@pytest.mark.parametrize('use_case', [UseCase.CLASSIFICATION_BINARY, UseCase.CLASSIFICATION_MULTICLASS])
def test_metric_factory_raises_invalid_args_exception_when_key_unknown(use_case):  # noqa: D103
    with pytest.raises(InvalidArgumentsException):
        _ = MetricFactory.create('foo', use_case)


@pytest.mark.parametrize('use_case', [UseCase.CLASSIFICATION_BINARY, UseCase.CLASSIFICATION_MULTICLASS])
def test_metric_factory_raises_invalid_args_exception_when_invalid_key_type_given(use_case):  # noqa: D103
    with pytest.raises(InvalidArgumentsException):
        _ = MetricFactory.create(123, use_case)
