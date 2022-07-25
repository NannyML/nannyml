# Author:   Niels Nuyttens  <niels@nannyml.com>
#
# License: Apache Software License 2.0

"""Tests for the chunking functionality."""
import datetime
import math
from typing import List

import numpy as np
import pandas as pd
import pytest
from pandas import Timestamp

from nannyml.chunk import Chunk, Chunker, CountBasedChunker, DefaultChunker, SizeBasedChunker
from nannyml.exceptions import ChunkerException, InvalidArgumentsException

rng = np.random.default_rng()


@pytest.fixture
def sample_chunk() -> Chunk:  # noqa: D103
    df = pd.DataFrame(rng.uniform(0, 100, size=(100, 4)), columns=list('ABCD'))
    chunk = Chunk(key='key', data=df)
    chunk.period = 'reference'
    chunk.start_index = 0
    chunk.end_index = 100
    chunk.start_datetime = datetime.datetime.min
    chunk.end_datetime = datetime.datetime.max
    return chunk


@pytest.fixture
def sample_chunk_data() -> pd.DataFrame:  # noqa: D103
    data = pd.DataFrame(pd.date_range(start='1/6/2020', freq='10min', periods=20 * 1008), columns=['ordered_at'])
    data['week'] = data.ordered_at.dt.isocalendar().week - 1
    data['period'] = 'reference'
    data.loc[data.week >= 11, ['period']] = 'analysis'
    np.random.seed(13)
    data['f1'] = np.random.randn(data.shape[0])
    data['f2'] = np.random.rand(data.shape[0])
    data['f3'] = np.random.randint(4, size=data.shape[0])
    data['f4'] = np.random.randint(20, size=data.shape[0])
    data['y_pred'] = np.random.randint(2, size=data.shape[0])
    data['y_true'] = np.random.randint(2, size=data.shape[0])
    data['timestamp'] = data['ordered_at']

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


@pytest.mark.parametrize(
    'text',
    [
        'key=key',
        'data=pd.DataFrame[[100x4]]',
        'is_transition=False',
        'period=reference',
        f'start_datetime={datetime.datetime.min}',
        f'end_datetime={datetime.datetime.max}',
        'start_index=0',
        'end_index=100',
    ],
)
def test_chunk_repr_should_contain_attribute(sample_chunk, text):  # noqa: D103
    sut = str(sample_chunk)
    assert text in sut


def test_chunk_len_should_return_data_length(sample_chunk):  # noqa: D103
    sut = len(sample_chunk)
    assert sut == len(sample_chunk.data)


def test_chunk_len_should_return_0_for_empty_chunk():  # noqa: D103
    sut = len(Chunk(key='test', data=pd.DataFrame()))
    assert sut == 0


def test_chunker_should_log_warning_when_less_than_6_chunks(sample_chunk_data, caplog):  # noqa: D103
    class SimpleChunker(Chunker):
        def _split(self, data: pd.DataFrame, timestamp_column_name: str, minimum_chunk_size: int = None) -> List[Chunk]:
            return [Chunk(key='row0', data=data)]

    c = SimpleChunker()
    with pytest.warns(UserWarning, match="The resulting number of chunks is too low."):
        _ = c.split(sample_chunk_data, timestamp_column_name='timestamp')


def test_chunker_should_log_warning_when_some_chunks_are_underpopulated(sample_chunk_data, caplog):  # noqa: D103
    class SimpleChunker(Chunker):
        def _split(self, data: pd.DataFrame, timestamp_column_name: str, minimum_chunk_size: int = None) -> List[Chunk]:
            return [Chunk(key='row0', data=data.iloc[[0]])]

    c = SimpleChunker()
    with pytest.warns(UserWarning, match="The resulting list of chunks contains 1 underpopulated chunks."):
        _ = c.split(sample_chunk_data, minimum_chunk_size=100000, timestamp_column_name='timestamp')


def test_chunker_should_set_index_boundaries(sample_chunk_data):  # noqa: D103
    class SimpleChunker(Chunker):
        def _split(self, data: pd.DataFrame, timestamp_column_name: str, minimum_chunk_size: int = None) -> List[Chunk]:
            return [
                Chunk(key='[0:6665]', data=data.iloc[0:6666, :]),
                Chunk(key='[6666:13331]', data=data.iloc[6666:13332, :]),
                Chunk(key='[13332:20160]', data=data.iloc[13332:, :]),
            ]

    chunker = SimpleChunker()
    sut = chunker.split(data=sample_chunk_data, timestamp_column_name='timestamp')
    assert sut[0].start_index == 0
    assert sut[0].end_index == 6665
    assert sut[1].start_index == 6666
    assert sut[1].end_index == 13331
    assert sut[2].start_index == 13332
    assert sut[2].end_index == 20159


def test_chunker_should_include_all_data_columns_by_default(sample_chunk_data):  # noqa: D103
    class SimpleChunker(Chunker):
        def _split(self, data: pd.DataFrame, timestamp_column_name: str, minimum_chunk_size: int = None) -> List[Chunk]:
            return [Chunk(key='row0', data=data)]

    c = SimpleChunker()
    sut = c.split(sample_chunk_data, timestamp_column_name='timestamp')[0].data.columns
    assert sorted(sut) == sorted(sample_chunk_data.columns)


def test_chunker_should_only_include_listed_columns_when_given_columns_param(sample_chunk_data):  # noqa: D103
    class SimpleChunker(Chunker):
        def _split(self, data: pd.DataFrame, timestamp_column_name: str, minimum_chunk_size: int = None) -> List[Chunk]:
            return [Chunk(key='row0', data=data)]

    columns = ['f1', 'f3', 'period']
    c = SimpleChunker()
    sut = c.split(sample_chunk_data, columns=columns, timestamp_column_name='timestamp')[0].data.columns
    assert sorted(sut) == sorted(columns)


def test_chunker_should_raise_chunker_exception_upon_exception_during_inherited_split_execution(  # noqa: D103
    sample_chunk_data,
):
    class SimpleChunker(Chunker):
        def _split(self, data: pd.DataFrame, timestamp_column_name: str, minimum_chunk_size: int = None) -> List[Chunk]:
            raise RuntimeError("oops, I broke it again")

    c = SimpleChunker()
    with pytest.raises(ChunkerException):
        _ = c.split(sample_chunk_data, timestamp_column_name='timestamp')


def test_chunker_should_fail_when_timestamp_column_not_provided(sample_chunk_data):  # noqa: D103
    class SimpleChunker(Chunker):
        def _split(self, data: pd.DataFrame, timestamp_column_name: str, minimum_chunk_size: int = None) -> List[Chunk]:
            return [Chunk(key='row0', data=data)]

    c = SimpleChunker()
    with pytest.raises(TypeError, match="'timestamp_column_name'"):
        c.split(sample_chunk_data)


def test_chunker_should_fail_when_timestamp_column_is_not_present(sample_chunk_data):  # noqa: D103
    class SimpleChunker(Chunker):
        def _split(self, data: pd.DataFrame, timestamp_column_name: str, minimum_chunk_size: int = None) -> List[Chunk]:
            return [Chunk(key='row0', data=data)]

    c = SimpleChunker()
    with pytest.raises(InvalidArgumentsException, match="timestamp column 'foo' not in columns"):
        c.split(sample_chunk_data, timestamp_column_name='foo')


def test_size_based_chunker_raises_exception_when_passed_nan_size(sample_chunk_data):  # noqa: D103
    with pytest.raises(InvalidArgumentsException):
        _ = SizeBasedChunker(chunk_size='size?')


def test_size_based_chunker_raises_exception_when_passed_negative_size(sample_chunk_data):  # noqa: D103
    with pytest.raises(InvalidArgumentsException):
        _ = SizeBasedChunker(chunk_size=-1)


def test_size_based_chunker_raises_exception_when_passed_zero_size(sample_chunk_data):  # noqa: D103
    with pytest.raises(InvalidArgumentsException):
        _ = SizeBasedChunker(chunk_size=0)


def test_size_based_chunker_works_with_empty_dataset():  # noqa: D103
    chunker = SizeBasedChunker(chunk_size=100)
    sut = chunker.split(
        pd.DataFrame(columns=['date', 'timestamp', 'f1', 'f2', 'f3', 'f4']), timestamp_column_name='timestamp'
    )
    assert len(sut) == 0


def test_size_based_chunker_returns_chunks_of_required_size(sample_chunk_data):  # noqa: D103
    chunk_size = 1500
    chunker = SizeBasedChunker(chunk_size=chunk_size)
    sut = chunker.split(sample_chunk_data, timestamp_column_name='timestamp')
    assert len(sut[0]) == chunk_size
    assert len(sut) == math.ceil(sample_chunk_data.shape[0] / chunk_size)


def test_size_based_chunker_returns_last_chunk_that_is_partially_filled(sample_chunk_data):  # noqa: D103
    chunk_size = 3333
    expected_last_chunk_size = sample_chunk_data.shape[0] % chunk_size
    chunker = SizeBasedChunker(chunk_size)
    sut = chunker.split(sample_chunk_data, timestamp_column_name='timestamp')
    assert len(sut[-1]) == expected_last_chunk_size


def test_size_based_chunker_works_when_data_set_is_multiple_of_chunk_size(sample_chunk_data):
    chunk_size = 1000
    data = sample_chunk_data.loc[0:19999, :]
    chunker = SizeBasedChunker(chunk_size)
    sut = []
    try:
        sut = chunker.split(data, timestamp_column_name='timestamp')
    except Exception as exc:
        pytest.fail(f'an unexpected exception occurred: {exc}')

    assert len(sut[-1]) == chunk_size


def test_size_based_chunker_drops_last_incomplete_chunk_when_set_drop_incomplete_is_true(  # noqa: D103
    sample_chunk_data,
):
    chunk_size = 3333
    chunker = SizeBasedChunker(chunk_size, drop_incomplete=True)
    sut = chunker.split(sample_chunk_data, timestamp_column_name='timestamp')
    assert len(sut[-1]) == chunk_size


def test_size_based_chunker_uses_observations_to_set_chunk_date_boundaries(sample_chunk_data):  # noqa: D103
    chunker = SizeBasedChunker(chunk_size=5000)
    sut = chunker.split(sample_chunk_data, timestamp_column_name='timestamp')
    assert sut[0].start_datetime == Timestamp(year=2020, month=1, day=6, hour=0, minute=0, second=0)
    assert sut[-1].end_datetime == Timestamp(year=2020, month=5, day=24, hour=23, minute=50, second=0)


def test_size_based_chunker_assigns_observation_range_to_chunk_keys(sample_chunk_data):  # noqa: D103
    chunk_size = 1500
    last_chunk_start = (sample_chunk_data.shape[0] // chunk_size) * chunk_size
    last_chunk_end = sample_chunk_data.shape[0] - 1

    chunker = SizeBasedChunker(chunk_size=chunk_size)
    sut = chunker.split(sample_chunk_data, timestamp_column_name='timestamp')
    assert sut[0].key == '[0:1499]'
    assert sut[1].key == '[1500:2999]'
    assert sut[-1].key == f'[{last_chunk_start}:{last_chunk_end}]'


def test_count_based_chunker_raises_exception_when_passed_nan_size(sample_chunk_data):  # noqa: D103
    with pytest.raises(InvalidArgumentsException):
        _ = CountBasedChunker(chunk_count='size?')


def test_count_based_chunker_raises_exception_when_passed_negative_size(sample_chunk_data):  # noqa: D103
    with pytest.raises(InvalidArgumentsException):
        _ = CountBasedChunker(chunk_count=-1)


def test_count_based_chunker_raises_exception_when_passed_zero_size(sample_chunk_data):  # noqa: D103
    with pytest.raises(InvalidArgumentsException):
        _ = CountBasedChunker(chunk_count=0)


def test_count_based_chunker_works_with_empty_dataset():  # noqa: D103
    chunker = CountBasedChunker(chunk_count=5)
    sut = chunker.split(
        pd.DataFrame(columns=['date', 'timestamp', 'f1', 'f2', 'f3', 'f4']), timestamp_column_name='timestamp'
    )
    assert len(sut) == 0


def test_count_based_chunker_returns_chunks_of_required_size(sample_chunk_data):  # noqa: D103
    chunk_count = 5
    chunker = CountBasedChunker(chunk_count=chunk_count)
    sut = chunker.split(sample_chunk_data, timestamp_column_name='timestamp')
    assert len(sut[0]) == sample_chunk_data.shape[0] // chunk_count
    assert len(sut) == chunk_count


def test_count_based_chunker_uses_observations_to_set_chunk_date_boundaries(sample_chunk_data):  # noqa: D103
    chunker = CountBasedChunker(chunk_count=20)
    sut = chunker.split(sample_chunk_data, timestamp_column_name='timestamp')
    assert sut[0].start_datetime == Timestamp(year=2020, month=1, day=6, hour=0, minute=0, second=0)
    assert sut[-1].end_datetime == Timestamp(year=2020, month=5, day=24, hour=23, minute=50, second=0)


def test_count_based_chunker_assigns_observation_range_to_chunk_keys(sample_chunk_data):  # noqa: D103
    chunk_count = 5

    chunker = CountBasedChunker(chunk_count=chunk_count)
    sut = chunker.split(sample_chunk_data, timestamp_column_name='timestamp')
    assert sut[0].key == '[0:4031]'
    assert sut[1].key == '[4032:8063]'
    assert sut[-1].key == '[16128:20159]'


def test_default_chunker_splits_into_ten_chunks(sample_chunk_data):  # noqa: D103
    expected_size = sample_chunk_data.shape[0] / 10
    sut = DefaultChunker().split(sample_chunk_data, timestamp_column_name='timestamp')
    assert len(sut) == 10
    assert len(sut[0]) == expected_size
    assert len(sut[1]) == expected_size
    assert len(sut[-1]) == expected_size
