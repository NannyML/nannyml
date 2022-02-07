#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import abc
from typing import List

import pandas as pd

from nannyml.chunk import Chunk, Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.metadata import NML_METADATA_COLUMNS, ModelMetadata


class DriftCalculator(abc.ABC):
    """Base class for drift calculation."""

    def __init__(self):
        """Constructs a new DriftCalculator instance."""
        pass

    def calculate(
        self,
        reference_data: pd.DataFrame,
        analysis_data: pd.DataFrame,
        model_metadata: ModelMetadata,
        chunker: Chunker,
        features: List[str] = None,
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

    def calculate(
        self,
        reference_data: pd.DataFrame,
        analysis_data: pd.DataFrame,
        model_metadata: ModelMetadata,
        chunker: Chunker,
        features: List[str] = None,
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
        reference_data : DataFrame
            The reference data set
        analysis_data : DataFrame
            The analysis data set
        model_metadata : ModelMetadata
            The metadata describing both reference and analysis data sets
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
        if reference_data.empty:
            raise InvalidArgumentsException('reference dataset contains no rows. Provide a valid reference data set.')

        if analysis_data.empty:
            raise InvalidArgumentsException('analysis dataset contains no rows. Provide a valid reference data set.')

        # Create metadata columns, just to be sure.
        reference_data = model_metadata.enrich(reference_data)
        analysis_data = model_metadata.enrich(analysis_data)

        # Generate chunks

        if not features:
            features = [f.name for f in model_metadata.features]
        features = NML_METADATA_COLUMNS + features

        # TODO verify: throw them together? Chunk twice?
        reference_chunks = chunker.split(reference_data, columns=features)
        analysis_chunks = chunker.split(analysis_data, columns=features)

        # Alternatively: chunk them together, then split afterwards
        # chunks = chunker.split(reference_features.append(analysis_features))
        # reference_chunks = [c for c in chunks if c.partition == 'reference' or c.is_transition]
        # analysis_chunks = [c for c in chunks if c.partition == 'analysis' or c.is_transition]

        return self._calculate_drift(
            reference_chunks=reference_chunks, analysis_chunks=analysis_chunks, model_metadata=model_metadata
        )

    def _calculate_drift(
        self, reference_chunks: List[Chunk], analysis_chunks: List[Chunk], model_metadata: ModelMetadata
    ) -> pd.DataFrame:
        raise NotImplementedError
