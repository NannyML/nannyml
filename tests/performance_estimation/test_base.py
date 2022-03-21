#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit tests for performance estimation."""

from typing import Tuple

import pandas as pd
import pytest

from nannyml.chunk import DefaultChunker, PeriodBasedChunker, SizeBasedChunker  # , _minimum_chunk_size
from nannyml.datasets import load_synthetic_sample
from nannyml.exceptions import InvalidArgumentsException
from nannyml.metadata import NML_METADATA_COLUMNS, ModelMetadata, extract_metadata
from nannyml.performance_estimation import BasePerformanceEstimator
from nannyml.performance_estimation.base import PerformanceEstimatorResult


@pytest.fixture
def sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, _ = load_synthetic_sample()
    return ref_df, ana_df


@pytest.fixture
def sample_metadata(sample_data) -> ModelMetadata:  # noqa: D103
    md = extract_metadata(sample_data[0])
    md.target_column_name = 'work_home_actual'
    return md


@pytest.fixture
def base_estimator(sample_metadata) -> BasePerformanceEstimator:  # noqa: D103
    return BasePerformanceEstimator(model_metadata=sample_metadata, chunk_size=5000)


@pytest.fixture
def simple_estimator(sample_metadata) -> BasePerformanceEstimator:  # noqa: D103
    return SimpleEstimator(model_metadata=sample_metadata, chunk_size=5000)


class SimpleEstimator(BasePerformanceEstimator):  # noqa: D101
    def _fit(self, reference_data: pd.DataFrame):
        pass

    def _estimate(self, data: pd.DataFrame) -> PerformanceEstimatorResult:
        features_and_metadata = NML_METADATA_COLUMNS + self.selected_features
        chunks = self.chunker.split(data, columns=features_and_metadata, minimum_chunk_size=50)
        return PerformanceEstimatorResult(
            model_metadata=self.model_metadata,
            estimated_data=pd.DataFrame(columns=self.selected_features).assign(key=[chunk.key for chunk in chunks]),
        )


def test_base_estimator_given_empty_reference_data_should_raise_invalid_args_exception(  # noqa: D103
    base_estimator, sample_data
):
    empty_ref_data = pd.DataFrame(columns=sample_data[0].columns)
    with pytest.raises(InvalidArgumentsException):
        base_estimator.fit(empty_ref_data)


def test_base_estimator_given_empty_analysis_data_should_raise_invalid_args_exception(  # noqa: D103
    base_estimator, sample_data
):
    with pytest.raises(InvalidArgumentsException):
        base_estimator.estimate(data=pd.DataFrame(columns=sample_data[1].columns))


def test_base_estimator_given_empty_features_list_should_calculate_for_all_features(  # noqa: D103
    simple_estimator, sample_data, sample_metadata
):
    ref_data, ana_data = sample_data
    simple_estimator.fit(ref_data)
    sut = simple_estimator.estimate(data=ana_data)

    assert len(sut.data.columns) == len(sample_metadata.features) + 1
    for f in sample_metadata.features:
        assert f.column_name in sut.data.columns


def test_base_estimator_given_non_empty_features_list_only_calculates_for_these_features(  # noqa: D103
    sample_data, sample_metadata
):
    simple_estimator = SimpleEstimator(
        sample_metadata, chunk_size=5000, features=['salary_range', 'distance_from_office']
    )
    ref_data, ana_data = sample_data
    simple_estimator.fit(ref_data)
    sut = simple_estimator.estimate(data=ana_data)

    assert len(sut.data.columns) == 3
    assert 'key' in sut.data.columns
    assert 'salary_range' in sut.data.columns
    assert 'distance_from_office' in sut.data.columns


def test_base_estimator_uses_size_based_chunker_when_given_chunk_size(sample_data, sample_metadata):  # noqa: D103
    simple_estimator = SimpleEstimator(sample_metadata, chunk_size=1000)
    simple_estimator.fit(sample_data[0])
    sut = simple_estimator.estimate(sample_data[1]).data['key']
    expected = [
        c.key for c in SizeBasedChunker(1000).split(sample_metadata.enrich(sample_data[1]), minimum_chunk_size=1)
    ]

    assert len(expected) == len(sut)
    assert sorted(expected) == sorted(sut)


def test_base_estimator_uses_count_based_chunker_when_given_chunk_number(sample_data, sample_metadata):  # noqa: D103
    simple_estimator = SimpleEstimator(sample_metadata, chunk_number=100)
    simple_estimator.fit(sample_data[0])
    sut = simple_estimator.estimate(sample_data[1]).data['key']

    assert 100 == len(sut)


def test_base_estimator_uses_period_based_chunker_when_given_chunk_period(sample_data, sample_metadata):  # noqa: D103
    simple_estimator = SimpleEstimator(sample_metadata, chunk_period='W')
    simple_estimator.fit(sample_data[0])
    sut = simple_estimator.estimate(sample_data[1]).data['key']

    expected = [
        c.key
        for c in PeriodBasedChunker(offset='W').split(sample_metadata.enrich(sample_data[1]), minimum_chunk_size=1)
    ]

    assert len(sut) == len(expected)


# @pytest.mark.skip('should confirm default minimum chunk size to test this')
def test_base_estimator_uses_default_chunker_when_no_chunker_specified(sample_data, sample_metadata):  # noqa: D103
    simple_estimator = SimpleEstimator(sample_metadata)
    simple_estimator.fit(sample_data[0])
    sut = simple_estimator.estimate(sample_data[1]).data['key']

    expected = [c.key for c in DefaultChunker().split(sample_metadata.enrich(sample_data[1]), minimum_chunk_size=50)]

    assert len(expected) == len(sut)
    assert sorted(expected) == sorted(sut)
