#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit tests for performance estimation."""

from typing import Tuple

import pandas as pd
import pytest

from nannyml.base import AbstractEstimator, AbstractEstimatorResult
from nannyml.chunk import CountBasedChunker  # , _minimum_chunk_size
from nannyml.chunk import DefaultChunker, PeriodBasedChunker, SizeBasedChunker
from nannyml.datasets import load_synthetic_binary_classification_dataset


@pytest.fixture
def sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, _ = load_synthetic_binary_classification_dataset()
    return ref_df, ana_df


@pytest.fixture
def simple_estimator() -> AbstractEstimator:  # noqa: D103
    return SimpleEstimator(chunk_size=5000)


class SimpleEstimatorResult(AbstractEstimatorResult):
    def __init__(self, results_data, calculator):
        super().__init__(results_data)
        self.calculator = calculator

    @property
    def estimator_name(self) -> str:
        return 'simple_estimator'

    def plot(self):
        pass


class SimpleEstimator(AbstractEstimator):  # noqa: D101
    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs):  # noqa: D102
        return self

    def _estimate(self, data: pd.DataFrame, *args, **kwargs) -> SimpleEstimatorResult:  # noqa: D102
        chunks = self.chunker.split(data, timestamp_column_name='timestamp', minimum_chunk_size=50)
        return SimpleEstimatorResult(
            results_data=pd.DataFrame(columns=data.columns).assign(key=[chunk.key for chunk in chunks]),
            calculator=self,
        )


def test_base_estimator_uses_size_based_chunker_when_given_chunk_size():  # noqa: D103
    simple_estimator = SimpleEstimator(chunk_size=1000)
    assert isinstance(simple_estimator.chunker, SizeBasedChunker)
    assert simple_estimator.chunker.chunk_size == 1000


def test_base_estimator_uses_count_based_chunker_when_given_chunk_number():  # noqa: D103
    simple_estimator = SimpleEstimator(chunk_number=100)
    assert isinstance(simple_estimator.chunker, CountBasedChunker)
    assert simple_estimator.chunker.chunk_count == 100


def test_base_estimator_uses_period_based_chunker_when_given_chunk_period():  # noqa: D103
    simple_estimator = SimpleEstimator(chunk_period='W')
    assert isinstance(simple_estimator.chunker, PeriodBasedChunker)
    assert simple_estimator.chunker.offset == 'W'


def test_base_estimator_uses_default_chunker_when_no_chunker_specified():  # noqa: D103
    simple_estimator = SimpleEstimator()
    assert isinstance(simple_estimator.chunker, DefaultChunker)
