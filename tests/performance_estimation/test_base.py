#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit tests for performance estimation."""

from typing import List, Tuple

import pandas as pd
import pytest

from nannyml.chunk import Chunk, DefaultChunker, PeriodBasedChunker, SizeBasedChunker  # , _minimum_chunk_size
from nannyml.datasets import load_synthetic_sample
from nannyml.exceptions import InvalidArgumentsException, NotFittedException
from nannyml.metadata import ModelMetadata, extract_metadata
from nannyml.performance_estimation import BasePerformanceEstimator


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

    def _estimate(self, chunks: List[Chunk]) -> pd.DataFrame:
        return pd.DataFrame(columns=self.selected_features).assign(key=[chunk.key for chunk in chunks])


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

    assert len(sut.columns) == len(sample_metadata.features) + 1
    for f in sample_metadata.features:
        assert f.column_name in sut.columns


def test_base_estimator_given_non_empty_features_list_only_calculates_for_these_features(  # noqa: D103
    sample_data, sample_metadata
):
    simple_estimator = SimpleEstimator(
        sample_metadata, chunk_size=5000, features=['salary_range', 'distance_from_office']
    )
    ref_data, ana_data = sample_data
    simple_estimator.fit(ref_data)
    sut = simple_estimator.estimate(data=ana_data)

    assert len(sut.columns) == 3
    assert 'key' in sut.columns
    assert 'salary_range' in sut.columns
    assert 'distance_from_office' in sut.columns


def test_base_estimator_raises_calculator_not_fitted_exc_when_calculating_with_none_chunker(  # noqa: D103
    simple_estimator, sample_data, sample_metadata
):
    with pytest.raises(NotFittedException, match='chunker has not been set.'):
        _ = simple_estimator.estimate(data=sample_data[0])


def test_base_estimator_uses_size_based_chunker_when_given_chunk_size(sample_data, sample_metadata):  # noqa: D103
    simple_estimator = SimpleEstimator(sample_metadata, chunk_size=1000)
    simple_estimator.fit(sample_data[0])
    sut = simple_estimator.estimate(sample_data[1])['key']
    expected = [
        c.key for c in SizeBasedChunker(1000, minimum_chunk_size=1).split(sample_metadata.enrich(sample_data[1]))
    ]

    assert len(expected) == len(sut)
    assert sorted(expected) == sorted(sut)


def test_base_estimator_uses_count_based_chunker_when_given_chunk_number(sample_data, sample_metadata):  # noqa: D103
    simple_estimator = SimpleEstimator(sample_metadata, chunk_number=100)
    simple_estimator.fit(sample_data[0])
    sut = simple_estimator.estimate(sample_data[1])['key']

    assert 100 == len(sut)


def test_base_estimator_uses_period_based_chunker_when_given_chunk_period(sample_data, sample_metadata):  # noqa: D103
    simple_estimator = SimpleEstimator(sample_metadata, chunk_period='W')
    simple_estimator.fit(sample_data[0])
    sut = simple_estimator.estimate(sample_data[1])['key']

    expected = [
        c.key
        for c in PeriodBasedChunker(offset='W', minimum_chunk_size=1).split(sample_metadata.enrich(sample_data[1]))
    ]

    assert len(sut) == len(expected)


def test_base_estimator_uses_default_chunker_when_no_chunker_specified(sample_data, sample_metadata):  # noqa: D103
    simple_estimator = SimpleEstimator(sample_metadata)
    simple_estimator.fit(sample_data[0])
    sut = simple_estimator.estimate(sample_data[1])['key']

    expected = [c.key for c in DefaultChunker(minimum_chunk_size=500).split(sample_metadata.enrich(sample_data[1]))]

    assert len(expected) == len(sut)
    assert sorted(expected) == sorted(sut)


# TODO: Move test for CBPE
# def test_base_estimator_uses_default_chunker_when_no_chunker_specified(sample_data, sample_metadata):  # noqa: D103
#     simple_estimator = SimpleEstimator(sample_metadata)
#     simple_estimator.fit(sample_data[0])
#     sut = simple_estimator.estimate(sample_data[1])['key']

#     min_chunk_size = _minimum_chunk_size(data=sample_metadata.enrich(sample_data[0]))
#     expected = [
#         c.key for c in DefaultChunker(minimum_chunk_size=min_chunk_size).split(sample_metadata.enrich(sample_data[1]))
#     ]

#     assert len(expected) == len(sut)
#     assert sorted(expected) == sorted(sut)
