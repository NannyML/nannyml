#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit testing for CBPE."""

from typing import Tuple

import pandas as pd
import pytest

from nannyml import CBPE
from nannyml.chunk import DefaultChunker, _minimum_chunk_size
from nannyml.datasets import load_synthetic_sample

# from nannyml.exceptions import InvalidArgumentsException, NotFittedException
from nannyml.metadata import ModelMetadata, extract_metadata


@pytest.fixture
def sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, _ = load_synthetic_sample()
    return ref_df, ana_df


@pytest.fixture
def sample_metadata(sample_data) -> ModelMetadata:  # noqa: D103
    md = extract_metadata(sample_data[0])
    md.target_column_name = 'work_home_actual'
    return md


def test_cbpe_estimator_uses_default_chunker_when_no_chunker_specified(sample_data, sample_metadata):  # noqa: D103
    simple_estimator = CBPE(sample_metadata)
    simple_estimator.fit(sample_data[0])
    sut = simple_estimator.estimate(sample_data[1]).data['key']

    print('debug')
    print(sut)

    min_chunk_size = _minimum_chunk_size(data=sample_metadata.enrich(sample_data[0]))
    expected = [
        c.key for c in DefaultChunker().split(sample_metadata.enrich(sample_data[1]), minimum_chunk_size=min_chunk_size)
    ]

    assert len(expected) == len(sut)
    assert sorted(expected) == sorted(sut)
