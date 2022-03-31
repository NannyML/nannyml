#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing base classes for performance calculation."""

from typing import Dict, List, Union

import numpy as np
import pandas as pd

from nannyml import Chunker, InvalidArgumentsException, ModelMetadata
from nannyml.chunk import Chunk, CountBasedChunker, DefaultChunker, PeriodBasedChunker, SizeBasedChunker
from nannyml.exceptions import CalculatorNotFittedException
from nannyml.metadata import NML_METADATA_COLUMNS, NML_METADATA_PARTITION_COLUMN_NAME, NML_METADATA_TARGET_COLUMN_NAME
from nannyml.performance_calculation.metrics import Metric, MetricFactory
from nannyml.performance_calculation.result import PerformanceCalculatorResult
from nannyml.preprocessing import preprocess

TARGET_COMPLETENESS_RATE_COLUMN_NAME = 'NML_TARGET_INCOMPLETE'


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
        self._minimum_chunk_size = None

        if chunker is None:
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

        for metric in self.metrics:
            metric.fit(reference_data, self.chunker)

        self._minimum_chunk_size = np.max([metric.minimum_chunk_size() for metric in self.metrics])

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
        chunks = self.chunker.split(data, columns=features_and_metadata, minimum_chunk_size=self._minimum_chunk_size)

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
        metrics_results = {}
        for metric in self.metrics:
            chunk_metric = metric.calculate(chunk.data)
            metrics_results[metric.column_name] = chunk_metric
            metrics_results[f'{metric.column_name}_thresholds'] = (metric.lower_threshold, metric.upper_threshold)
            metrics_results[f'{metric.column_name}_alert'] = (
                metric.lower_threshold > chunk_metric or chunk_metric > metric.upper_threshold
            ) and (chunk.data[NML_METADATA_PARTITION_COLUMN_NAME] == 'analysis').all()
        return metrics_results
