#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing base classes for performance calculation."""

import abc
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from nannyml import Chunker, InvalidArgumentsException, ModelMetadata
from nannyml.chunk import (
    Chunk,
    CountBasedChunker,
    DefaultChunker,
    PeriodBasedChunker,
    SizeBasedChunker,
    _minimum_chunk_size,
)
from nannyml.exceptions import CalculatorNotFittedException
from nannyml.metadata import NML_METADATA_COLUMNS, NML_METADATA_PREDICTION_COLUMN_NAME, NML_METADATA_TARGET_COLUMN_NAME
from nannyml.performance_calculation.metrics import Metric, MetricFactory
from nannyml.preprocessing import preprocess

TARGET_COMPLETENESS_RATE_COLUMN_NAME = 'NML_TARGET_INCOMPLETE'


class PerformanceCalculatorResult(abc.ABC):
    """Contains the results of performance calculation and adds plotting functionality."""

    def __init__(
        self,
        performance_data: pd.DataFrame,
        model_metadata: ModelMetadata,
    ):
        """Creates a new PerformanceCalculatorResult instance.

        Parameters
        ----------
        performance_data : pd.DataFrame
            The results of the performance calculation.
        model_metadata :
            The metadata describing the monitored model.
        """
        self.data = performance_data
        self.metadata = model_metadata


class PerformanceCalculator:
    """Base class for performance metric calculation."""

    def __init__(
        self,
        model_metadata: ModelMetadata,
        metrics: List[Union[str, Metric]],
        chunk_size: int = None,
        chunk_number: int = None,
        chunk_period: str = None,
        chunker: Chunker = None,
    ):
        """Creates a new performance calculator.

        Parameters
        ----------
        model_metadata : ModelMetadata
            The metadata describing the monitored model.
        metrics: List[Union[str, Callable, Metric]]
            A list of metrics to calculate. These can be specified as a reference string, a function that performs
            the metric calculation or a Metric object that allows configuring the display name or thresholds.
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
        self.metadata = model_metadata
        self.metrics = [MetricFactory.create(m) for m in metrics]

        self.chunker = chunker
        self._chunk_size = chunk_size
        self._chunk_number = chunk_number
        self._chunk_period = chunk_period

    def fit(self, reference_data: pd.DataFrame):
        """Fits the calculator on the reference data, calibrating it for further use on the full dataset.

        Parameters
        ----------
        reference_data: pd.DataFrame
            Reference data for the model, i.e. model inputs and predictions enriched with target data.
        """
        if reference_data.empty:
            raise InvalidArgumentsException('reference data contains no rows. Provide a valid reference data set.')
        reference_data = preprocess(data=reference_data, model_metadata=self.metadata)

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

    def calculate(self, analysis_data: pd.DataFrame) -> PerformanceCalculatorResult:
        """Calculates performance on the analysis data, using the metrics specified on calculator creation.

        Parameters
        ----------
        analysis_data: pd.DataFrame
            Analysis data for the model, i.e. model inputs and predictions.
        """
        if analysis_data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        if self.metadata.target_column_name not in analysis_data.columns:
            raise InvalidArgumentsException(
                f"data does not contain target data column '{self.metadata.target_column_name}'."
            )

        # Preprocess data
        data: pd.DataFrame = preprocess(data=analysis_data, model_metadata=self.metadata)

        # Setup for target completeness rate
        data['NML_TARGET_INCOMPLETE'] = data[NML_METADATA_TARGET_COLUMN_NAME].isna().astype(np.int16)

        # Generate chunks
        features_and_metadata = NML_METADATA_COLUMNS + [TARGET_COMPLETENESS_RATE_COLUMN_NAME]
        if self.chunker is None:
            raise CalculatorNotFittedException(
                'chunker has not been set. '
                'Please ensure you run ``calculator.fit()`` '
                'before running ``calculator.calculate()``'
            )
        chunks = self.chunker.split(data, columns=features_and_metadata)

        # Construct result frame
        res = pd.DataFrame.from_records(
            [
                {
                    'key': chunk.key,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'start_date': chunk.start_datetime,
                    'end_date': chunk.end_datetime,
                    'partition': 'analysis' if chunk.is_transition else chunk.partition,
                    'targets_missing_rate': chunk.data[TARGET_COMPLETENESS_RATE_COLUMN_NAME].sum()
                    / chunk.data[TARGET_COMPLETENESS_RATE_COLUMN_NAME].count(),
                    **self._calculate_metrics_for_chunk(chunk),
                }
                for chunk in chunks
            ]
        )

        return PerformanceCalculatorResult(performance_data=res, model_metadata=self.metadata)

    def _calculate_metrics_for_chunk(self, chunk: Chunk) -> Dict:
        return {
            m.display_name: m.calculation_function(
                y_true=chunk.data[NML_METADATA_TARGET_COLUMN_NAME],
                y_pred=chunk.data[NML_METADATA_PREDICTION_COLUMN_NAME],
            )
            for m in self.metrics
        }
