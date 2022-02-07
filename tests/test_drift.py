#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Tests for Drift package."""

from typing import List

import numpy as np
import pandas as pd
import pytest

from nannyml.chunk import Chunk, CountBasedChunker, PeriodBasedChunker, SizeBasedChunker
from nannyml.drift import BaseDriftCalculator
from nannyml.drift.statistical_drift_calculator import StatisticalDriftCalculator
from nannyml.exceptions import InvalidArgumentsException
from nannyml.metadata import NML_METADATA_COLUMNS, ModelMetadata, extract_metadata


@pytest.fixture
def sample_drift_data() -> pd.DataFrame:  # noqa: D103
    data = pd.DataFrame(pd.date_range(start='1/6/2020', freq='10min', periods=20 * 1008), columns=['timestamp'])
    data['week'] = data.timestamp.dt.isocalendar().week - 1
    data['partition'] = 'reference'
    data.loc[data.week >= 11, ['partition']] = 'analysis'
    # data[NML_METADATA_PARTITION_COLUMN_NAME] = data['partition']  # simulate preprocessing
    np.random.seed(167)
    data['f1'] = np.random.randn(data.shape[0])
    data['f2'] = np.random.rand(data.shape[0])
    data['f3'] = np.random.randint(4, size=data.shape[0])
    data['f4'] = np.random.randint(20, size=data.shape[0])
    data['output'] = np.random.randint(2, size=data.shape[0])
    data['actual'] = np.random.randint(2, size=data.shape[0])

    # Rule 1b is the shifted feature, 75% 0 instead of 50%
    rule1a = {2: 0, 3: 1}
    rule1b = {2: 0, 3: 0}
    data.loc[data.week < 16, ['f3']] = data.loc[data.week < 16, ['f3']].replace(rule1a)
    data.loc[data.week >= 16, ['f3']] = data.loc[data.week >= 16, ['f3']].replace(rule1b)

    # Rule 2b is the shifted feature
    c1 = 'white'
    c2 = 'red'
    c3 = 'green'
    c4 = 'blue'

    rule2a = {
        0: c1,
        1: c1,
        2: c1,
        3: c1,
        4: c1,
        5: c2,
        6: c2,
        7: c2,
        8: c2,
        9: c2,
        10: c3,
        11: c3,
        12: c3,
        13: c3,
        14: c3,
        15: c4,
        16: c4,
        17: c4,
        18: c4,
        19: c4,
    }

    rule2b = {
        0: c1,
        1: c1,
        2: c1,
        3: c1,
        4: c1,
        5: c2,
        6: c2,
        7: c2,
        8: c2,
        9: c2,
        10: c3,
        11: c3,
        12: c3,
        13: c1,
        14: c1,
        15: c4,
        16: c4,
        17: c4,
        18: c1,
        19: c2,
    }

    data.loc[data.week < 16, ['f4']] = data.loc[data.week < 16, ['f4']].replace(rule2a)
    data.loc[data.week >= 16, ['f4']] = data.loc[data.week >= 16, ['f4']].replace(rule2b)

    data.loc[data.week >= 16, ['f1']] = data.loc[data.week >= 16, ['f1']] + 0.6
    data.loc[data.week >= 16, ['f2']] = np.sqrt(data.loc[data.week >= 16, ['f2']])
    data['id'] = data.index
    data.drop(columns=['week'], inplace=True)

    return data


@pytest.fixture
def sample_drift_metadata(sample_drift_data):  # noqa: D103
    return extract_metadata(sample_drift_data, model_name='model')


class SimpleDriftCalculator(BaseDriftCalculator):
    """Dummy DriftCalculator implementation that returns a DataFrame with the selected feature columns, no rows."""

    def _calculate_drift(
        self, reference_chunks: List[Chunk], analysis_chunks: List[Chunk], model_metadata: ModelMetadata
    ) -> pd.DataFrame:
        df = analysis_chunks[0].data.drop(columns=NML_METADATA_COLUMNS)
        return pd.DataFrame(columns=df.columns)


def test_base_drift_calculator_given_empty_reference_data_should_raise_invalid_args_exception(  # noqa: D103
    sample_drift_data,
):
    calc = SimpleDriftCalculator()
    with pytest.raises(InvalidArgumentsException):
        calc.calculate(
            reference_data=pd.DataFrame(columns=sample_drift_data.columns),
            analysis_data=sample_drift_data,
            model_metadata=extract_metadata(sample_drift_data, model_name='model'),
            chunker=SizeBasedChunker(chunk_size=1000),
        )


def test_base_drift_calculator_given_empty_analysis_data_should_raise_invalid_args_exception(  # noqa: D103
    sample_drift_data,
):
    calc = SimpleDriftCalculator()
    with pytest.raises(InvalidArgumentsException):
        calc.calculate(
            reference_data=sample_drift_data,
            analysis_data=pd.DataFrame(columns=sample_drift_data.columns),
            model_metadata=extract_metadata(sample_drift_data, model_name='model'),
            chunker=SizeBasedChunker(chunk_size=1000),
        )


def test_base_drift_calculator_given_empty_features_list_should_calculate_for_all_features(  # noqa: D103
    sample_drift_data,
):
    calc = SimpleDriftCalculator()
    md = extract_metadata(sample_drift_data, model_name='model')
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    analysis_data = sample_drift_data.loc[sample_drift_data['partition'] == 'analysis']
    sut = calc.calculate(
        reference_data=ref_data,
        analysis_data=analysis_data,
        model_metadata=md,
        chunker=SizeBasedChunker(chunk_size=1000),
    )
    assert len(sut.columns) == len(md.features)
    for f in md.features:
        assert f.column_name in sut.columns


def test_base_drift_calculator_given_non_empty_features_list_should_only_calculate_for_these_features(  # noqa: D103
    sample_drift_data,
):
    calc = SimpleDriftCalculator()
    md = extract_metadata(sample_drift_data, model_name='model')
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    analysis_data = sample_drift_data.loc[sample_drift_data['partition'] == 'analysis']
    sut = calc.calculate(
        reference_data=ref_data,
        analysis_data=analysis_data,
        model_metadata=md,
        chunker=SizeBasedChunker(chunk_size=1000),
        features=['f1', 'f3'],
    )
    assert len(sut.columns) == 2
    assert 'f1' in sut.columns
    assert 'f3' in sut.columns


@pytest.mark.parametrize(
    'chunker',
    [
        (PeriodBasedChunker(offset='W')),
        (PeriodBasedChunker(offset='M')),
        (SizeBasedChunker(chunk_size=1000)),
        CountBasedChunker(chunk_count=25),
    ],
    ids=['chunk_period_weekly', 'chunk_period_monthly', 'chunk_size_1000', 'chunk_count_25'],
)
def test_statistical_drift_calculator_should_return_a_row_for_each_analysis_chunk_key(  # noqa: D103
    sample_drift_data, sample_drift_metadata, chunker
):
    calc = StatisticalDriftCalculator()
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    analysis_data = sample_drift_data.loc[sample_drift_data['partition'] == 'analysis']
    sut = calc.calculate(
        reference_data=ref_data,
        analysis_data=analysis_data,
        model_metadata=sample_drift_metadata,
        chunker=chunker,
    )

    chunks = chunker.split(sample_drift_metadata.enrich(analysis_data))
    assert len(chunks) == sut.shape[0]
    chunk_keys = [c.key for c in chunks]
    assert sorted(chunk_keys) == sorted(sut['chunk'].values)


@pytest.mark.skip("Awaiting proper static data")
def test_statistical_drift_calculator(sample_drift_data, sample_drift_metadata):  # noqa: D103
    calc = StatisticalDriftCalculator()
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    analysis_data = sample_drift_data.loc[sample_drift_data['partition'] == 'analysis']
    sut = calc.calculate(
        reference_data=ref_data,
        analysis_data=analysis_data,
        model_metadata=sample_drift_metadata,
        chunker=PeriodBasedChunker(offset='W'),
    )
    print(sut)
