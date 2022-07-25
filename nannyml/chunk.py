# Author:   Niels Nuyttens  <niels@nannyml.com>
#           Jakub Bialek    <jakub@nannyml.com>
#
# License: Apache Software License 2.0

"""NannyML module providing intelligent splitting of data into chunks."""

import abc
import logging
import warnings
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from dateutil.parser import ParserError  # type: ignore
from pandas import Period

from nannyml.exceptions import ChunkerException, InvalidArgumentsException

logger = logging.getLogger(__name__)


class Chunk:
    """A subset of data that acts as a logical unit during calculations."""

    def __init__(
        self,
        key: str,
        data: pd.DataFrame,
        start_datetime: datetime = datetime.max,
        end_datetime: datetime = datetime.max,
        period: str = None,
    ):
        """Creates a new chunk.

        Parameters
        ----------
        key : str, required.
            A value describing what data is wrapped in this chunk.
        data : DataFrame, required
            The data to be contained within the chunk.
        start_datetime: datetime
            The starting point in time for this chunk.
        end_datetime: datetime
            The end point in time for this chunk.
        period : string, optional
            The 'period' this chunk belongs to, for example 'reference' or 'analysis'.
        """
        self.key = key
        self.data = data
        self.period = period

        self.is_transition: bool = False

        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.start_index: int = 0
        self.end_index: int = 0

    def __repr__(self):
        """Returns textual summary of a chunk.

        Returns
        -------
        chunk_str: str

        """
        return (
            f'Chunk[key={self.key}, data=pd.DataFrame[[{self.data.shape[0]}x{self.data.shape[1]}]], '
            f'period={self.period}, is_transition={self.is_transition},'
            f'start_datetime={self.start_datetime}, end_datetime={self.end_datetime},'
            f'start_index={self.start_index}, end_index={self.end_index}]'
        )

    def __len__(self):
        """Returns the number of rows held within this chunk.

        Returns
        -------
        length: int
            Number of rows in the `data` property of the chunk.

        """
        return self.data.shape[0]


def _get_boundary_indices(c: Chunk):
    return c.data.index.min(), c.data.index.max()


class Chunker(abc.ABC):
    """Base class for Chunker implementations.

    Inheriting classes will split a DataFrame into a list of Chunks.
    They will do this based on several constraints, e.g. observation timestamps, number of observations per Chunk
    or a preferred number of Chunks.
    """

    def __init__(self):
        """Creates a new Chunker."""
        pass

    def split(
        self,
        data: pd.DataFrame,
        timestamp_column_name: str,
        columns=None,
        minimum_chunk_size: int = None,
    ) -> List[Chunk]:
        """Splits a given data frame into a list of chunks.

        This method provides a uniform interface across Chunker implementations to keep them interchangeable.

        After performing the implementation-specific `_split` method, there are some checks on the resulting chunk list.

        If the total number of chunks is low a warning will be written out to the logs.

        We dynamically determine the optimal minimum number of observations per chunk and then check if the resulting
        chunks contain at least as many. If there are any underpopulated chunks a warning will be written out in
        the logs.

        Parameters
        ----------
        data: DataFrame
            The data to be split into chunks
        timestamp_column_name: str
            Name of the column containing the timestamp of an observation.
        period_column_name: str
            Name of the column containing the period of an observation, if any.
        columns: List[str], default=None
            A list of columns to be included in the resulting chunk data. Unlisted columns will be dropped.
        minimum_chunk_size: int, default=None
            The recommended minimum number of observations a :class:`~nannyml.chunk.Chunk` should hold.
            When specified a warning will appear if the split results in underpopulated chunks.
            When not specified there will be no checks for underpopulated chunks.

        Returns
        -------
        chunks: List[Chunk]
            The list of chunks

        """
        if timestamp_column_name not in data.columns:
            raise InvalidArgumentsException(
                f"timestamp column '{timestamp_column_name}' not in columns: {list(data.columns)}."
            )

        data = data.sort_values(by=[timestamp_column_name]).reset_index(drop=True)

        try:
            chunks = self._split(data, timestamp_column_name, minimum_chunk_size)
        except Exception as exc:
            raise ChunkerException(f"could not split data into chunks: {exc}")

        for c in chunks:
            c.start_index, c.end_index = _get_boundary_indices(c)

            if columns is not None:
                c.data = c.data[columns]

        if len(chunks) < 6:
            # TODO wording
            warnings.warn(
                'The resulting number of chunks is too low. '
                'Please consider splitting your data in a different way or continue at your own risk.'
            )

        # check if all chunk sizes > minimal chunk size. If not, render a warning message.
        if minimum_chunk_size:
            underpopulated_chunks = [c for c in chunks if len(c) < minimum_chunk_size]

            if len(underpopulated_chunks) > 0:
                # TODO wording
                warnings.warn(
                    f'The resulting list of chunks contains {len(underpopulated_chunks)} underpopulated chunks. '
                    'They contain too few records to be statistically robust and might negatively influence '
                    'the quality of calculations. '
                    'Please consider splitting your data in a different way or continue at your own risk.'
                )

        return chunks

    # TODO wording
    @abc.abstractmethod
    def _split(self, data: pd.DataFrame, timestamp_column_name: str, minimum_chunk_size: int = None) -> List[Chunk]:
        """Splits the DataFrame into chunks.

        Abstract method, to be implemented within inheriting classes.

        Parameters
        ----------
        data: pandas.DataFrame
            The full dataset that should be split into Chunks
        minimum_chunk_size: int, default=None
            The recommended minimum number of observations a :class:`~nannyml.chunk.Chunk` should hold.

        Returns
        -------
        chunks: array of Chunks
            The array of Chunks after splitting the original DataFrame `data`

        See Also
        --------
        PeriodBasedChunker: Splits data based on the timestamp of observations
        SizeBasedChunker: Splits data based on the amount of observations in a Chunk
        CountBasedChunker: Splits data based on the resulting number of Chunks

        Notes
        -----
        There is a minimal number of observations that a Chunk should contain in order to retain statistical relevance.
        A chunker will log a warning message when your splitting criteria would result in underpopulated chunks.
        Note that in this situation calculation results may not be relevant.

        """
        pass  # pragma: no cover


class ChunkerFactory:
    @classmethod
    def get_chunker(
        cls,
        chunk_size: int = None,
        chunk_number: int = None,
        chunk_period: str = None,
        chunker: Chunker = None,
    ) -> Chunker:
        if chunker is not None:
            return chunker
        if chunk_size:
            return SizeBasedChunker(chunk_size=chunk_size)  # type: ignore
        elif chunk_number:
            return CountBasedChunker(chunk_count=chunk_number)  # type: ignore
        elif chunk_period:
            return PeriodBasedChunker(offset=chunk_period)  # type: ignore
        else:
            return DefaultChunker()  # type: ignore


class PeriodBasedChunker(Chunker):
    """A Chunker that will split data into Chunks based on a date column in the data.

    Examples
    --------
    Chunk using monthly periods and providing a column name

    >>> from nannyml.chunk import PeriodBasedChunker
    >>> df = pd.read_parquet('/path/to/my/data.pq')
    >>> chunker = PeriodBasedChunker(date_column_name='observation_date', offset='M')
    >>> chunks = chunker.split(data=df)

    Or chunk using weekly periods

    >>> from nannyml.chunk import PeriodBasedChunker
    >>> df = pd.read_parquet('/path/to/my/data.pq')
    >>> chunker = PeriodBasedChunker(date_column=df['observation_date'], offset='W', minimum_chunk_size=50)
    >>> chunks = chunker.split(data=df)

    """

    def __init__(
        self,
        offset: str = 'W',
    ):
        """Creates a new PeriodBasedChunker.

        Parameters
        ----------
        offset: a frequency string representing a pandas.tseries.offsets.DateOffset
            The offset determines how the time-based grouping will occur. A list of possible values
            is to be found at https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases.

        Returns
        -------
        chunker: a PeriodBasedChunker instance used to split data into time-based Chunks.
        """
        super().__init__()

        self.offset = offset

    def _split(self, data: pd.DataFrame, timestamp_column_name: str, minimum_chunk_size: int = None) -> List[Chunk]:
        chunks = []
        try:
            grouped_data = data.groupby(pd.to_datetime(data[timestamp_column_name]).dt.to_period(self.offset))

            k: Period
            for k in grouped_data.groups.keys():
                chunk = Chunk(
                    key=str(k), data=grouped_data.get_group(k), start_datetime=k.start_time, end_datetime=k.end_time
                )
                chunks.append(chunk)
        except KeyError:
            raise ChunkerException(f"could not find date_column '{timestamp_column_name}' in given data")

        except ParserError:
            raise ChunkerException(
                f"could not parse date_column '{timestamp_column_name}' values as dates."
                f"Please verify if you've specified the correct date column."
            )

        return chunks


class SizeBasedChunker(Chunker):
    """A Chunker that will split data into Chunks based on the preferred number of observations per Chunk.

    Notes
    -----
    - Chunks are adjacent, not overlapping
    - There will be no "incomplete chunks", so the leftover observations that cannot fill an entire chunk will
      be dropped by default.

    Examples
    --------
    Chunk using monthly periods and providing a column name

    >>> from nannyml.chunk import SizeBasedChunker
    >>> df = pd.read_parquet('/path/to/my/data.pq')
    >>> chunker = SizeBasedChunker(chunk_size=2000, minimum_chunk_size=50)
    >>> chunks = chunker.split(data=df)

    """

    def __init__(self, chunk_size: int, drop_incomplete: bool = False):
        """Create a new SizeBasedChunker.

        Parameters
        ----------
        chunk_size: int
            The preferred size of the resulting Chunks, i.e. the number of observations in each Chunk.
        drop_incomplete: bool, default=False
            Indicates whether the final Chunk after splitting should be dropped if it doesn't contain
            ``chunk_size`` observations. Defaults to ``False``, i.e. the final chunk will always be kept.

        Returns
        -------
        chunker: a size-based instance used to split data into Chunks of a constant size.

        """
        super().__init__()

        # TODO wording
        if not isinstance(chunk_size, (int, np.int64)):
            raise InvalidArgumentsException(
                f"given chunk_size is of type {type(chunk_size)} but should be an int."
                f"Please provide an integer as a chunk size"
            )

        # TODO wording
        if chunk_size <= 0:
            raise InvalidArgumentsException(
                f"given chunk_size {chunk_size} is less then or equal to zero."
                f"The chunk size should always be larger then zero"
            )

        self.chunk_size = chunk_size
        self.drop_incomplete = drop_incomplete

    def _split(self, data: pd.DataFrame, timestamp_column_name: str, minimum_chunk_size: int = None) -> List[Chunk]:
        def _create_chunk(index: int, data: pd.DataFrame, chunk_size: int) -> Chunk:
            chunk_data = data.loc[index : index + chunk_size - 1, :]
            min_date = pd.to_datetime(chunk_data[timestamp_column_name].min())
            max_date = pd.to_datetime(chunk_data[timestamp_column_name].max())
            return Chunk(
                key=f'[{index}:{index + chunk_size - 1}]',
                data=chunk_data,
                start_datetime=min_date,
                end_datetime=max_date,
            )

        data = data.copy().reset_index(drop=True)
        chunks = [
            _create_chunk(index=i, data=data, chunk_size=self.chunk_size)
            for i in range(0, len(data), self.chunk_size)
            if i + self.chunk_size - 1 < len(data)
        ]

        if not self.drop_incomplete and (data.shape[0] % self.chunk_size != 0):
            chunks += [
                _create_chunk(
                    index=self.chunk_size * (data.shape[0] // self.chunk_size),
                    data=data,
                    chunk_size=(data.shape[0] % self.chunk_size),
                )
            ]

        return chunks


class CountBasedChunker(Chunker):
    """A Chunker that will split data into chunks based on the preferred number of total chunks.

    Examples
    --------
    >>> from nannyml.chunk import CountBasedChunker
    >>> df = pd.read_parquet('/path/to/my/data.pq')
    >>> chunker = CountBasedChunker(chunk_count=100, minimum_chunk_size=50)
    >>> chunks = chunker.split(data=df)

    """

    def __init__(self, chunk_count: int):
        """Creates a new CountBasedChunker.

        It will calculate the amount of observations per chunk based on the given chunk count.
        It then continues to split the data into chunks just like a SizeBasedChunker does.

        Parameters
        ----------
        chunk_count: int
            The amount of chunks to split the data in.

        Returns
        -------
        chunker: CountBasedChunker

        """
        super().__init__()

        # TODO wording
        if not isinstance(chunk_count, int):
            raise InvalidArgumentsException(
                f"given chunk_count is of type {type(chunk_count)} but should be an int."
                f"Please provide an integer as a chunk count"
            )

        # TODO wording
        if chunk_count <= 0:
            raise InvalidArgumentsException(
                f"given chunk_count {chunk_count} is less then or equal to zero."
                f"The chunk count should always be larger then zero"
            )

        self.chunk_count = chunk_count

    def _split(self, data: pd.DataFrame, timestamp_column_name: str, minimum_chunk_size: int = None) -> List[Chunk]:
        if data.shape[0] == 0:
            return []

        data = data.copy().reset_index()

        chunk_size = data.shape[0] // self.chunk_count
        chunks = SizeBasedChunker(chunk_size=chunk_size).split(
            data=data, timestamp_column_name=timestamp_column_name, minimum_chunk_size=minimum_chunk_size
        )
        return chunks


class DefaultChunker(Chunker):
    """Splits data into about 10 chunks.

    Examples
    --------
    >>> from nannyml.chunk import DefaultChunker
    >>> df = pd.read_parquet('/path/to/my/data.pq')
    >>> chunker = DefaultChunker(minimum_chunk_size=50)
    >>> chunks = chunker.split(data=df)
    """

    DEFAULT_CHUNK_COUNT = 10

    def __init__(self):
        """Creates a new DefaultChunker."""
        super(DefaultChunker, self).__init__()

    def _split(self, data: pd.DataFrame, timestamp_column_name: str, minimum_chunk_size: int = None) -> List[Chunk]:
        if data.shape[0] == 0:
            return []

        data = data.copy().reset_index(drop=True)

        chunk_size = data.shape[0] // self.DEFAULT_CHUNK_COUNT
        chunks = SizeBasedChunker(chunk_size=chunk_size).split(
            data=data,
            timestamp_column_name=timestamp_column_name,
            minimum_chunk_size=minimum_chunk_size,
        )
        return chunks
