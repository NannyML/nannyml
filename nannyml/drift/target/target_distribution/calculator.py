#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module for target distribution monitoring."""
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from nannyml.chunk import Chunker, CountBasedChunker, DefaultChunker, PeriodBasedChunker, SizeBasedChunker
from nannyml.drift.target.target_distribution.result import TargetDistributionResult
from nannyml.exceptions import InvalidArgumentsException
from nannyml.metadata import (
    NML_METADATA_COLUMNS,
    NML_METADATA_PARTITION_COLUMN_NAME,
    NML_METADATA_TARGET_COLUMN_NAME,
    ModelMetadata,
)
from nannyml.preprocessing import preprocess


class TargetDistributionCalculator:
    """Calculates target distribution for a given dataset."""

    def __init__(
        self,
        model_metadata: ModelMetadata,
        chunk_size: int = None,
        chunk_number: int = None,
        chunk_period: str = None,
        chunker: Chunker = None,
    ):
        """Constructs a new TargetDistributionCalculator.

        Parameters
        ----------
        model_metadata: ModelMetadata
            Metadata for the model whose data is to be processed.
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

        self._reference_targets: pd.Series = None  # type: ignore

        # TODO: determine better min_chunk_size for target distribution
        self._minimum_chunk_size = 300

    def fit(self, reference_data: pd.DataFrame):
        """Fits the calculator to reference data.

        During fitting the reference target data is validated and stored for later use.
        """
        if reference_data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        if self.metadata.target_column_name not in reference_data.columns:
            raise InvalidArgumentsException(
                f"data does not contain target data column '{self.metadata.target_column_name}'."
            )

        self._reference_targets = preprocess(data=reference_data, model_metadata=self.metadata)[
            NML_METADATA_TARGET_COLUMN_NAME
        ]

    def calculate(self, data: pd.DataFrame):
        """Calculates the target distribution of a binary classifier.

        Parameters
        ----------
        data: pd.DataFrame
            Data for the model, i.e. model inputs, predictions and targets.
        """
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        if self.metadata.target_column_name not in data.columns:
            raise InvalidArgumentsException(
                f"data does not contain target data column '{self.metadata.target_column_name}'."
            )

        # Preprocess data
        data = preprocess(data=data, model_metadata=self.metadata)

        data['NML_TARGET_INCOMPLETE'] = data[NML_METADATA_TARGET_COLUMN_NAME].isna().astype(np.int16)

        # Generate chunks
        features_and_metadata = NML_METADATA_COLUMNS + ['NML_TARGET_INCOMPLETE']
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
                    'targets_missing_rate': (
                        chunk.data['NML_TARGET_INCOMPLETE'].sum() / chunk.data['NML_TARGET_INCOMPLETE'].count()
                    ),
                    **_calculate_target_drift_for_chunk(self._reference_targets, chunk.data),
                }
                for chunk in chunks
            ]
        )

        return TargetDistributionResult(target_distribution=res, model_metadata=self.metadata)


def _calculate_target_drift_for_chunk(reference_targets: pd.Series, data: pd.DataFrame) -> Dict:
    targets = data[NML_METADATA_TARGET_COLUMN_NAME]
    statistic, p_value, _, _ = chi2_contingency(
        pd.concat([reference_targets.value_counts(), targets.value_counts()], axis=1).fillna(0)
    )

    _ALERT_THRESHOLD_P_VALUE = 0.05

    is_analysis = 'analysis' in set(data[NML_METADATA_PARTITION_COLUMN_NAME].unique())

    return {
        'metric_target_drift': targets.mean(),
        'statistical_target_drift': statistic,
        'p_value': p_value,
        'thresholds': _ALERT_THRESHOLD_P_VALUE,
        'alert': (p_value < _ALERT_THRESHOLD_P_VALUE) and is_analysis,
        'significant': p_value < _ALERT_THRESHOLD_P_VALUE,
    }
