# Author:   Niels Nuyttens  <niels@nannyml.com>
#
# License: Apache Software License 2.0

import math
from typing import List

import numpy as np
import pandas as pd
import pytest

from nannyml._chunk import (
    NML_METADATA_PARTITION_COLUMN_NAME,
    Chunk,
    Chunker,
    CountBasedChunker,
    PeriodBasedChunker,
    SizeBasedChunker,
)
from nannyml.exceptions import ChunkerException, InvalidArgumentsException

rng = np.random.default_rng()


@pytest.fixture
def sample_chunk() -> Chunk:
    df = pd.DataFrame(rng.uniform(0, 100, size=(100, 4)), columns=list('ABCD'))
    return Chunk(key='key', data=df)


@pytest.fixture
def sample_chunk_data() -> pd.DataFrame:
    data = pd.DataFrame(pd.date_range(start='1/6/2020', freq='10min', periods=20 * 1008), columns=['ordered_at'])
    data['week'] = data.ordered_at.dt.isocalendar().week - 1
    data['partition'] = 'reference'
    data.loc[data.week >= 11, ['partition']] = 'analysis'
    data[NML_METADATA_PARTITION_COLUMN_NAME] = data['partition']  # simulate preprocessing
    np.random.seed(13)
    data['f1'] = np.random.randn(data.shape[0])
    data['f2'] = np.random.rand(data.shape[0])
    data['f3'] = np.random.randint(4, size=data.shape[0])
    data['f4'] = np.random.randint(20, size=data.shape[0])

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

    return data


def test_chunk_repr_should_contain_key(sample_chunk):
    sut = str(sample_chunk)
    assert 'key=key' in sut


def test_chunk_repr_should_contain_data(sample_chunk):
    sut = str(sample_chunk)
    assert 'data=pd.DataFrame[[100x4]]' in sut


def test_chunk_repr_should_contain_transition(sample_chunk):
    sut = str(sample_chunk)
    assert 'is_transition=False' in sut


def test_chunk_repr_should_contain_partition(sample_chunk):
    sut = str(sample_chunk)
    assert 'partition' in sut


def test_chunk_len_should_return_data_length(sample_chunk):
    sut = len(sample_chunk)
    assert sut == len(sample_chunk.data)


def test_chunk_len_should_return_0_for_empty_chunk():
    sut = len(Chunk(key='test', data=pd.DataFrame()))
    assert sut == 0


def test_chunker_should_log_warning_when_less_than_6_chunks(capsys, sample_chunk_data):
    class SimpleChunker(Chunker):
        def _split(self, data: pd.DataFrame) -> List[Chunk]:
            return [Chunk(key='row0', data=data)]

    c = SimpleChunker()
    _ = c.split(sample_chunk_data)
    sut, err = capsys.readouterr()

    # TODO: check if extra log config is required
    # assert "The resulting number of chunks is too low." in sut


def test_chunker_should_log_warning_when_some_chunks_are_underpopulated(capsys, sample_chunk_data):
    class SimpleChunker(Chunker):
        def _split(self, data: pd.DataFrame) -> List[Chunk]:
            return [Chunk(key='row0', data=data.iloc[[0]])]

    c = SimpleChunker()
    _ = c.split(sample_chunk_data)
    sut, err = capsys.readouterr()

    # TODO: check if extra log config is required
    # assert "The resulting list of chunks contains 1 underpopulated chunks." in sut


def test_chunker_should_set_chunk_transition_flag_when_it_contains_observations_from_multiple_partitions(
    sample_chunk_data,
):
    class SimpleChunker(Chunker):
        def _split(self, data: pd.DataFrame) -> List[Chunk]:
            return [
                Chunk(key='[0:6665]', data=data.iloc[0:6665, :]),
                Chunk(key='[6666:13331]', data=data.iloc[6666:13331, :]),
                Chunk(key='[13332:20160]', data=data.iloc[13332:, :]),
            ]

    chunker = SimpleChunker()
    sut = chunker.split(data=sample_chunk_data)

    assert len(sut) == 3
    assert sut[0].is_transition is False
    assert sut[2].is_transition is False
    assert sut[1].is_transition


def test_period_based_chunker_raises_exception_when_passed_neither_date_col_name_or_date_col(sample_chunk_data):
    with pytest.raises(InvalidArgumentsException):
        _ = PeriodBasedChunker()


def test_period_based_chunker_works_with_date_column_name(sample_chunk_data):
    chunker = PeriodBasedChunker(date_column_name='ordered_at')
    sut = chunker.split(sample_chunk_data)
    assert len(sut) == 20
    assert len(sut[0]) == 1008


def test_period_based_chunker_works_with_date_column(sample_chunk_data):
    chunker = PeriodBasedChunker(date_column=sample_chunk_data['ordered_at'])
    sut = chunker.split(sample_chunk_data)
    assert len(sut) == 20  # 20 weeks
    assert len(sut[0]) == 1008  # Every


def test_period_based_chunker_works_with_non_default_offset(sample_chunk_data):
    chunker = PeriodBasedChunker(date_column_name='ordered_at', offset='M')
    sut = chunker.split(sample_chunk_data)
    assert len(sut) == 5  # 20 weeks == 5 months


def test_period_based_chunker_works_with_empty_dataset():
    chunker = PeriodBasedChunker(date_column_name='date')
    sut = chunker.split(pd.DataFrame(columns=['date', 'f1', 'f2', 'f3', 'f4']))
    assert len(sut) == 0


def test_period_based_chunker_fails_when_date_column_does_not_exist(sample_chunk_data):
    chunker = PeriodBasedChunker(date_column_name='non_existent')
    with pytest.raises(ChunkerException, match="could not find date_column 'non_existent' in given data"):
        _ = chunker.split(sample_chunk_data)


def test_period_based_chunker_fails_when_date_column_does_not_contain_dates(sample_chunk_data):
    chunker = PeriodBasedChunker(date_column_name='f4')
    with pytest.raises(ChunkerException, match="could not parse date_column 'f4'"):
        _ = chunker.split(sample_chunk_data)


def test_period_based_chunker_assigns_periods_to_chunk_keys(sample_chunk_data):
    chunker = PeriodBasedChunker(date_column_name='ordered_at', offset='M')
    sut = chunker.split(sample_chunk_data)
    assert sut[0].key == '2020-01'
    assert sut[1].key == '2020-02'
    assert sut[-1].key == '2020-05'


def test_size_based_chunker_raises_exception_when_passed_nan_size(sample_chunk_data):
    with pytest.raises(InvalidArgumentsException):
        _ = SizeBasedChunker(chunk_size='size?')


def test_size_based_chunker_raises_exception_when_passed_negative_size(sample_chunk_data):
    with pytest.raises(InvalidArgumentsException):
        _ = SizeBasedChunker(chunk_size=-1)


def test_size_based_chunker_raises_exception_when_passed_zero_size(sample_chunk_data):
    with pytest.raises(InvalidArgumentsException):
        _ = SizeBasedChunker(chunk_size=0)


def test_size_based_chunker_works_with_empty_dataset():
    chunker = SizeBasedChunker(chunk_size=100)
    sut = chunker.split(pd.DataFrame(columns=['date', 'f1', 'f2', 'f3', 'f4']))
    assert len(sut) == 0


def test_size_based_chunker_returns_chunks_of_required_size(sample_chunk_data):
    chunk_size = 1500
    chunker = SizeBasedChunker(chunk_size=chunk_size)
    sut = chunker.split(sample_chunk_data)
    assert len(sut[0]) == chunk_size
    assert len(sut) == sample_chunk_data.shape[0] // chunk_size


def test_size_based_chunker_assigns_observation_range_to_chunk_keys(sample_chunk_data):
    chunk_size = 1500
    last_chunk_start = (math.floor(sample_chunk_data.shape[0] / chunk_size) - 1) * chunk_size
    last_chunk_end = math.floor(sample_chunk_data.shape[0] / chunk_size) * chunk_size - 1

    chunker = SizeBasedChunker(chunk_size=chunk_size)
    sut = chunker.split(sample_chunk_data)
    assert sut[0].key == '[0:1499]'
    assert sut[1].key == '[1500:2999]'
    assert sut[-1].key == f'[{last_chunk_start}:{last_chunk_end}]'


def test_count_based_chunker_raises_exception_when_passed_nan_size(sample_chunk_data):
    with pytest.raises(InvalidArgumentsException):
        _ = CountBasedChunker(chunk_count='size?')


def test_count_based_chunker_raises_exception_when_passed_negative_size(sample_chunk_data):
    with pytest.raises(InvalidArgumentsException):
        _ = CountBasedChunker(chunk_count=-1)


def test_count_based_chunker_raises_exception_when_passed_zero_size(sample_chunk_data):
    with pytest.raises(InvalidArgumentsException):
        _ = CountBasedChunker(chunk_count=0)


def test_count_based_chunker_works_with_empty_dataset():
    chunker = CountBasedChunker(chunk_count=5)
    sut = chunker.split(pd.DataFrame(columns=['date', 'f1', 'f2', 'f3', 'f4']))
    assert len(sut) == 0


def test_count_based_chunker_returns_chunks_of_required_size(sample_chunk_data):
    chunk_count = 5
    chunker = CountBasedChunker(chunk_count=chunk_count)
    sut = chunker.split(sample_chunk_data)
    assert len(sut[0]) == sample_chunk_data.shape[0] // chunk_count
    assert len(sut) == chunk_count


def test_count_based_chunker_assigns_observation_range_to_chunk_keys(sample_chunk_data):
    chunk_count = 5

    chunker = CountBasedChunker(chunk_count=chunk_count)
    sut = chunker.split(sample_chunk_data)
    assert sut[0].key == '[0:4031]'
    assert sut[1].key == '[4032:8063]'
    assert sut[-1].key == '[16128:20159]'
