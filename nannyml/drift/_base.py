#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import abc
from typing import List

import pandas as pd
import pandas.tseries.offsets

from nannyml.chunk import Chunk, Chunker, CountBasedChunker, DefaultChunker, PeriodBasedChunker, SizeBasedChunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.metadata import NML_METADATA_COLUMNS, ModelMetadata
from nannyml.preprocessing import preprocess


class DriftCalculator(abc.ABC):
    """Base class for drift calculation."""

    def __init__(self, model_metadata: ModelMetadata, features: List[str] = None):
        """Constructs a new DriftCalculator instance."""
        self.model_metadata = model_metadata
        if not features:
            features = [f.column_name for f in self.model_metadata.features]
        self.selected_features = features

    def fit(self, reference_data: pd.DataFrame):
        """Fits the calculator on the reference data, calibrating it for further use on the full dataset."""
        raise NotImplementedError

    def calculate(
        self,
        data: pd.DataFrame,
        chunk_size: int = None,
        chunk_number: int = None,
        chunk_period: pandas.tseries.offsets.DateOffset = None,
        chunker: Chunker = None,
    ) -> pd.DataFrame:
        """Executes the drift calculation.

        NannyML will use the model metadata to provide additional information about the features.
        You can select the features included in the calculation by using the `features` parameter.

        """
        raise NotImplementedError


class BaseDriftCalculator(DriftCalculator, abc.ABC):
    """Abstract class with a basic implementation of drift calculation.

    This class provides a `calculate` function that will take care of all the data preparations that need to occur
    before handing off the actual calculation to inheriting classes by overriding the `_calculate_drift` method.

    """

    def fit(self, reference_data: pd.DataFrame):
        """Calibrates a DriftCalculator using a reference dataset.

        Parameters
        ----------
        reference_data : pd.DataFrame
            The reference data used to calibrate the DriftCalculator.
        """
        if reference_data.empty:
            raise InvalidArgumentsException('reference data contains no rows. Provide a valid reference data set.')
        reference_data = preprocess(data=reference_data, model_metadata=self.model_metadata)
        self._fit(reference_data)

    def _fit(self, reference_data: pd.DataFrame):
        raise NotImplementedError

    def calculate(
        self,
        data: pd.DataFrame,
        chunk_size: int = None,
        chunk_number: int = None,
        chunk_period: str = None,
        chunker: Chunker = None,
    ) -> pd.DataFrame:
        """Performs validations and transformations before delegating the calculation to implementing classes.

        Steps taken in this function are:

        - Creating fixed metadata columns in both analysis and reference data sets
        - Filtering for only features listed in the `features` parameter (if any given)
        - Basic validations on both data sets
        - Generating chunks from both sets
        - Calling the `_calculate_drift` function, providing a list of reference data chunks and a list of analysis
          data chunks.

        Parameters
        ----------
        data : DataFrame
            The data to be analyzed
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
        features : List[str], default=None
            If any feature column names are given here the drift calculation will only happen for those features.
            When not specified the calculation will occur for all features.

        Returns
        -------
        data_drifts : DataFrame
            A DataFrame where a cell contains the drift that occurred for a specific feature (current column)
            when comparing a Chunk of analysis data (current row) to a corresponding Chunk of reference data.

            The results thus contain a row for each Chunk in the analysis set and a column for each feature identified
            in the model metadata (or less if filtered using the `features` parameter).

        """
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Provide a valid data set.')

        if chunker is None:
            if chunk_size:
                chunker = SizeBasedChunker(chunk_size=chunk_size)
            elif chunk_number:
                chunker = CountBasedChunker(chunk_count=chunk_number)
            elif chunk_period:
                chunker = PeriodBasedChunker(offset=chunk_period)
            else:
                chunker = DefaultChunker()

        # Preprocess data
        data = preprocess(data=data, model_metadata=self.model_metadata)

        # Generate chunks
        features_and_metadata = NML_METADATA_COLUMNS + self.selected_features
        chunks = chunker.split(data, columns=features_and_metadata)

        return self._calculate_drift(chunks=chunks)

    def _calculate_drift(
        self,
        chunks: List[Chunk],
    ) -> pd.DataFrame:
        raise NotImplementedError
