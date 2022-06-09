#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit tests for performance metrics."""
from typing import Any, Dict, List, Tuple

import pandas as pd
import pytest

from nannyml.datasets import (
    load_synthetic_binary_classification_dataset,
    load_synthetic_multiclass_classification_dataset,
)
from nannyml.exceptions import InvalidArgumentsException, MissingMetadataException
from nannyml.metadata import BinaryClassificationMetadata, ModelMetadata, ModelType, MulticlassClassificationMetadata
from nannyml.metadata.extraction import extract_metadata
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


@pytest.fixture
def binary_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, tgt_df = load_synthetic_binary_classification_dataset()
    ref_df['y_pred'] = ref_df['y_pred_proba'].map(lambda p: p >= 0.8).astype(int)
    ana_df['y_pred'] = ana_df['y_pred_proba'].map(lambda p: p >= 0.8).astype(int)

    return ref_df, ana_df, tgt_df


@pytest.fixture
def binary_metadata(binary_data) -> ModelMetadata:  # noqa: D103
    md = extract_metadata(binary_data[0], model_type='classification_binary')
    md.target_column_name = 'work_home_actual'
    return md


@pytest.fixture
def multiclass_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, tgt_df = load_synthetic_multiclass_classification_dataset()

    return ref_df, ana_df, tgt_df


@pytest.fixture
def multiclass_metadata(multiclass_data) -> ModelMetadata:  # noqa: D103
    md = extract_metadata(multiclass_data[0], model_type='classification_multiclass')
    return md


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


# cannot combine fixtures and parametrization in pytest, so need to split binary/multiclass cases
@pytest.mark.parametrize('metric', ['roc_auc', 'f1', 'precision', 'recall', 'specificity', 'accuracy'])
def test_metric_calculation_raises_invalid_metadata_exception_when_missing_binary_target_metadata(  # noqa: D103
    metric, binary_data, binary_metadata
):
    binary_metadata.target_column_name = None
    m = MetricFactory.create(key=metric, metadata=binary_metadata)
    with pytest.raises(MissingMetadataException):
        # m.fit(binary_data[0], chunker=DefaultChunker())  # not really required, but for completeness
        m.calculate(binary_data[1])


@pytest.mark.parametrize('metric', ['roc_auc', 'f1', 'precision', 'recall', 'specificity', 'accuracy'])
def test_metric_calculation_raises_invalid_metadata_exception_when_missing_target_multiclass_metadata(  # noqa: D103
    metric, multiclass_data, multiclass_metadata
):
    multiclass_metadata.target_column_name = None
    m = MetricFactory.create(key=metric, metadata=multiclass_metadata)
    with pytest.raises(MissingMetadataException):
        # m.fit(multiclass_data[0], chunker=DefaultChunker())  # not really required, but for completeness
        m.calculate(multiclass_data[1])


# cannot combine fixtures and parametrization in pytest, so need to split binary/multiclass cases
@pytest.mark.parametrize('metric', ['f1', 'precision', 'recall', 'specificity', 'accuracy'])
def test_metric_calculation_raises_invalid_metadata_exception_when_missing_binary_prediction_metadata(  # noqa: D103
    metric, binary_data, binary_metadata
):
    binary_metadata.prediction_column_name = None
    m = MetricFactory.create(key=metric, metadata=binary_metadata)
    with pytest.raises(MissingMetadataException):
        # m.fit(binary_data[0], chunker=DefaultChunker())  # not really required, but for completeness
        m.calculate(binary_data[1])


@pytest.mark.parametrize('metric', ['f1', 'precision', 'recall', 'specificity', 'accuracy'])
def test_metric_calculation_raises_invalid_metadata_exception_when_missing_multiclass_target_metadata(  # noqa: D103
    metric, multiclass_data, multiclass_metadata
):
    multiclass_metadata.prediction_column_name = None
    m = MetricFactory.create(key=metric, metadata=multiclass_metadata)
    with pytest.raises(MissingMetadataException):
        # m.fit(multiclass_data[0], chunker=DefaultChunker())  # not really required, but for completeness
        m.calculate(multiclass_data[1])


def test_metric_roc_auc_calculation_raises_invalid_metadata_exception_when_missing_binary_score_metadata(  # noqa: D103
    binary_data, binary_metadata
):
    binary_metadata.predicted_probability_column_name = None
    m = MetricFactory.create(key='roc_auc', metadata=binary_metadata)
    with pytest.raises(MissingMetadataException):
        # m.fit(binary_data[0], chunker=DefaultChunker())  # not really required, but for completeness
        m.calculate(binary_data[1])


def test_metric_roc_auc_calculation_raises_invalid_metadata_exception_when_missing_mc_score_metadata(  # noqa: D103
    multiclass_data, multiclass_metadata
):
    multiclass_metadata.predicted_probabilities_column_names = None
    m = MetricFactory.create(key='roc_auc', metadata=multiclass_metadata)
    with pytest.raises(MissingMetadataException):
        # m.fit(binary_data[0], chunker=DefaultChunker())  # not really required, but for completeness
        m.calculate(multiclass_data[1])
