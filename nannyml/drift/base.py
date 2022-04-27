#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing base classes for drift calculation."""
from __future__ import annotations

import abc
from typing import List

import pandas as pd
import plotly.graph_objects

from nannyml.chunk import Chunk, Chunker, CountBasedChunker, DefaultChunker, PeriodBasedChunker, SizeBasedChunker
from nannyml.metadata.base import ModelMetadata


class DriftResult(abc.ABC):
    """Contains the results of a drift calculation and provides additional functionality such as plotting.

    The result of the :meth:`~nannyml.drift.base.DriftCalculator.calculate` method of a
    :class:`~nannyml.drift.base.DriftCalculator`.

    It is an abstract class containing shared properties and methods across implementations.
    For each :class:`~nannyml.drift.base.DriftCalculator` class there will be an associated
    :class:`~nannyml.drift.base.DriftResult` implementation.
    """

    def __init__(self, analysis_data: List[Chunk], drift_data: pd.DataFrame, model_metadata: ModelMetadata):
        """Creates a new DriftResult instance.

        Parameters
        ----------
        analysis_data: List[Chunk]
            The data that was provided to calculate drift on. This is required in order to plot distributions.
        drift_data: pd.DataFrame
            The results of the drift calculation.
        model_metadata: ModelMetadata
            The metadata describing the monitored model. Used to
        """
        self._analysis_data = analysis_data
        self.data = drift_data.copy(deep=True)
        self.metadata = model_metadata

    def plot(self, *args, **kwargs) -> plotly.graph_objects.Figure:
        """Plot drift results."""
        raise NotImplementedError


class DriftCalculator(abc.ABC):
    """Base class for drift calculation."""

    def __init__(
        self,
        model_metadata: ModelMetadata,
        features: List[str] = None,
        chunk_size: int = None,
        chunk_number: int = None,
        chunk_period: str = None,
        chunker: Chunker = None,
    ):
        """Creates a new instance of an abstract DriftCalculator.

        Parameters
        ----------
        model_metadata: ModelMetadata
            Metadata telling the DriftCalculator what columns are required for drift calculation.
        features: List[str]
            An optional list of feature column names. When set only these columns will be included in the
            drift calculation. If not set it will default to all feature column names and the model prediction.
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
        self.model_metadata = model_metadata
        if not features:
            features = [f.column_name for f in self.model_metadata.features]
        self.selected_features = features

        if chunker is None:
            # Note:
            # minimum chunk size is only needed if a chunker with a user specified minimum chunk size is not provided
            if chunk_size:
                self.chunker = SizeBasedChunker(chunk_size=chunk_size)  # type: ignore
            elif chunk_number:
                self.chunker = CountBasedChunker(chunk_count=chunk_number)  # type: ignore
            elif chunk_period:
                self.chunker = PeriodBasedChunker(offset=chunk_period)  # type: ignore
            else:
                self.chunker = DefaultChunker()  # type: ignore
        else:
            self.chunker = chunker  # type: ignore

    def fit(self, reference_data: pd.DataFrame) -> DriftCalculator:
        """Fits the calculator on the reference data, calibrating it for further use on the full dataset."""
        raise NotImplementedError

    def calculate(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Executes the drift calculation.

        NannyML will use the model metadata to provide additional information about the features.
        You can select the features included in the calculation by using the `features` parameter.

        """
        raise NotImplementedError
