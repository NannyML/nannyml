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
from nannyml.drift._base import ChunkerPreset, _preset_to_chunker
from nannyml.drift.reconstruction_error_drift_calcutor import ReconstructionErrorDriftCalculator
from nannyml.drift.statistical_drift_calculator import StatisticalDriftCalculator, calculate_statistical_drift
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
        self,
        reference_data: pd.DataFrame,
        chunks: List[Chunk],
        model_metadata: ModelMetadata,
        selected_features: List[str],
    ) -> pd.DataFrame:
        df = chunks[0].data.drop(columns=NML_METADATA_COLUMNS)
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
    'preset,expected',
    [
        ('size_1000', SizeBasedChunker(chunk_size=1000)),
        ('period_week', PeriodBasedChunker(offset='W')),
        ('period_month', PeriodBasedChunker(offset='M')),
        ('period_day', PeriodBasedChunker(offset='D')),
        ('count_100', CountBasedChunker(chunk_count=100)),
    ],
)
def test_preset_to_chunker_returns_correct_chunker_given_string_value(preset, expected):  # noqa: D103
    sut = _preset_to_chunker(preset)
    assert isinstance(sut, type(expected))


@pytest.mark.parametrize(
    'preset,expected',
    [
        (ChunkerPreset.SIZE_1000, SizeBasedChunker(chunk_size=1000)),
        (ChunkerPreset.PERIOD_WEEK, PeriodBasedChunker(offset='W')),
        (ChunkerPreset.PERIOD_MONTH, PeriodBasedChunker(offset='M')),
        (ChunkerPreset.PERIOD_DAY, PeriodBasedChunker(offset='D')),
        (ChunkerPreset.COUNT_100, CountBasedChunker(chunk_count=100)),
    ],
)
def test_preset_to_chunker_returns_correct_chunker_given_chunker_preset(preset, expected):  # noqa: D103
    sut = _preset_to_chunker(preset)
    assert isinstance(sut, type(expected))


def test_preset_to_chunker_raises_exception_when_unknown_preset_given():  # noqa: D103
    with pytest.raises(InvalidArgumentsException, match="unknown chunker preset value 'dunno'"):
        _ = _preset_to_chunker('dunno')


def test_base_drift_calculator_uses_default_size_1000_chunker_when_no_chunker_specified(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    class TestDriftCalculator(BaseDriftCalculator):
        def _calculate_drift(
            self,
            reference_data: pd.DataFrame,
            chunks: List[Chunk],
            model_metadata: ModelMetadata,
            selected_features: List[str],
        ) -> pd.DataFrame:
            chunk_keys = [c.key for c in chunks]
            return pd.DataFrame({'keys': chunk_keys})

    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    analysis_data = sample_drift_data.loc[sample_drift_data['partition'] == 'analysis']

    calc = TestDriftCalculator()
    sut = calc.calculate(ref_data, analysis_data, sample_drift_metadata)['keys']
    expected = [c.key for c in SizeBasedChunker(chunk_size=1000).split(sample_drift_metadata.enrich(sample_drift_data))]

    assert len(expected) == len(sut)
    assert sorted(expected) == sorted(sut)


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

    chunks = chunker.split(sample_drift_metadata.enrich(sample_drift_data))
    assert len(chunks) == sut.shape[0]
    chunk_keys = [c.key for c in chunks]
    assert 'chunk' in sut.columns
    assert sorted(chunk_keys) == sorted(sut['chunk'].values)


def test_statistical_drift_calculator_should_return_a_stat_column_and_p_value_column_for_each_feature(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    calc = StatisticalDriftCalculator()
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    analysis_data = sample_drift_data.loc[sample_drift_data['partition'] == 'analysis']
    sut = calc.calculate(
        reference_data=ref_data,
        analysis_data=analysis_data,
        model_metadata=sample_drift_metadata,
        chunker=SizeBasedChunker(chunk_size=1000),
    ).columns

    for f in sample_drift_metadata.features:
        assert f'{f.column_name}_statistic' in sut
        assert f'{f.column_name}_p_value' in sut


def test_statistical_drift_calculator(sample_drift_data, sample_drift_metadata):  # noqa: D103
    calc = StatisticalDriftCalculator()
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    analysis_data = sample_drift_data.loc[sample_drift_data['partition'] == 'analysis']
    try:
        _ = calc.calculate(
            reference_data=ref_data,
            analysis_data=analysis_data,
            model_metadata=sample_drift_metadata,
            chunker=PeriodBasedChunker(offset='W'),
        )
    except Exception:
        pytest.fail()


def test_calculate_statistical_drift_function_runs_on_defaults(sample_drift_data, sample_drift_metadata):  # noqa: D103
    reference_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    analysis_data = sample_drift_data.loc[sample_drift_data['partition'] == 'analysis']
    try:
        calculate_statistical_drift(reference_data, analysis_data, sample_drift_metadata)
    except Exception:
        pytest.fail()


def test_reconstruction_error_drift_calculator(sample_drift_data, sample_drift_metadata):  # noqa: D103
    calc = ReconstructionErrorDriftCalculator(n_components=0.65)
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    analysis_data = sample_drift_data.loc[sample_drift_data['partition'] == 'analysis']
    try:
        drift = calc.calculate(
            reference_data=ref_data,
            analysis_data=analysis_data,
            model_metadata=sample_drift_metadata,
            chunker=PeriodBasedChunker(offset='W'),
        )
        print(drift)
    except Exception:
        pytest.fail()
