#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing base classes for performance estimation."""
import abc
from typing import List

import pandas as pd

from nannyml.chunk import (
    Chunk,
    Chunker,
    CountBasedChunker,
    DefaultChunker,
    PeriodBasedChunker,
    SizeBasedChunker,
    _minimum_chunk_size,
)
from nannyml.exceptions import InvalidArgumentsException, NotFittedException
from nannyml.metadata import NML_METADATA_COLUMNS, ModelMetadata
from nannyml.preprocessing import preprocess


class PerformanceEstimator(abc.ABC):
    """Abstract class for performance estimation."""

    def __init__(self, model_metadata: ModelMetadata, features: List[str] = None):
        """Creates a new instance of a performance estimator."""
        self.model_metadata = model_metadata
        if not features:
            features = [f.column_name for f in self.model_metadata.features]
        self.selected_features = features

    def fit(self, reference_data: pd.DataFrame):
        """Fits the data on a reference data set."""
        raise NotImplementedError

    def estimate(self, data: pd.DataFrame):
        """Estimate performance given a data set lacking ground truth."""
        raise NotImplementedError


class BasePerformanceEstimator(PerformanceEstimator):
    """Base class for performance estimation.

    Provides some boilerplate to deal with chunking and data preprocessing.
    """

    def __init__(
        self,
        model_metadata: ModelMetadata,
        features: List[str] = None,
        chunk_size: int = None,
        chunk_number: int = None,
        chunk_period: str = None,
        chunker: Chunker = None,
    ):
        """Creates a new BasePerformanceEstimator.

        Parameters
        ----------
        model_metadata: ModelMetadata
            Metadata telling the DriftCalculator what columns are required for drift calculation.
        features: List[str]
            An optional list of feature column names. When set only these columns will be included in the
            drift calculation. If not set it will default to all feature column names.
        chunk_size: int
            Splits the data into chunks containing `chunks_size` observations.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunk_number: int
            Splits the data into `chunk_number` pieces.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunk_period: str
            Splits the data according to the given period.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunker : Chunker
            The `Chunker` used to split the data sets into a lists of chunks.

        """
        super(BasePerformanceEstimator, self).__init__(model_metadata, features)

        self.chunker = chunker
        self._chunk_size = chunk_size
        self._chunk_number = chunk_number
        self._chunk_period = chunk_period

    def fit(self, reference_data: pd.DataFrame):
        """Fits the estimator on a reference data set.

        Parameters
        ----------
        reference_data : pd.DataFrame
            A data set for which performance is generally accepted as exemplary.
        """
        if reference_data.empty:
            raise InvalidArgumentsException('reference data contains no rows. Provide a valid reference data set.')
        reference_data = preprocess(data=reference_data, model_metadata=self.model_metadata)

        # Calculate minimum chunk size based on reference data (we need y_pred_proba and y_true for this)
        # Store for DefaultChunker init during calculation
        # TODO: refactor as factory function in chunk module
        minimum_chunk_size = _minimum_chunk_size(data=reference_data)
        if self.chunker is None:
            if self._chunk_size:
                self.chunker = SizeBasedChunker(chunk_size=self._chunk_size, minimum_chunk_size=minimum_chunk_size)
            elif self._chunk_number:
                self.chunker = CountBasedChunker(chunk_count=self._chunk_number, minimum_chunk_size=minimum_chunk_size)
            elif self._chunk_period:
                self.chunker = PeriodBasedChunker(offset=self._chunk_period, minimum_chunk_size=minimum_chunk_size)
            else:
                self.chunker = DefaultChunker(minimum_chunk_size=minimum_chunk_size)

        self._fit(reference_data)

    def _fit(self, reference_data: pd.DataFrame):
        raise NotImplementedError

    def estimate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Performs validations and transformations before delegating the estimation to implementing classes.

        Steps taken in this function are:

        - Creating fixed metadata columns in both analysis and reference data sets
        - Filtering only selected features
        - Splitting data into chunks
        - Calling the `_estimate` function

        Parameters
        ----------
        data : DataFrame
            The data to estimate performance for.

        Returns
        -------
        estimated_performance : DataFrame
            A DataFrame where a cell contains the estimated (overall) performance for each chunk.

        """
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Provide a valid data set.')

        # Preprocess data
        data = preprocess(data=data, model_metadata=self.model_metadata)

        # Generate chunks
        features_and_metadata = NML_METADATA_COLUMNS + self.selected_features
        if self.chunker is None:
            raise NotFittedException(
                'chunker has not been set. '
                'Please ensure you run ``estimator.fit()`` '
                'before running ``estimator.estimate()``'
            )
        chunks = self.chunker.split(data, columns=features_and_metadata)

        return self._estimate(chunks=chunks)

    def _estimate(self, chunks: List[Chunk]) -> pd.DataFrame:
        raise NotImplementedError
