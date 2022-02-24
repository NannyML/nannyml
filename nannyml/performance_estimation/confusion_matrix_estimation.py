#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Implementation of the CME estimator."""
from typing import List

import numpy as np
import pandas as pd

from nannyml import Chunk, Chunker, ModelMetadata
from nannyml.performance_estimation._base import BasePerformanceEstimator


class CME(BasePerformanceEstimator):
    """Performance estimator using the Confusion Matrix Estimation (CME) technique."""

    def __init__(
        self,
        model_metadata: ModelMetadata,
        features: List[str] = None,
        chunk_size: int = None,
        chunk_number: int = None,
        chunk_period: str = None,
        chunker: Chunker = None,
    ):
        """Creates a new CME performance estimator.

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
        super().__init__(model_metadata, features, chunk_size, chunk_number, chunk_period, chunker)

    def _fit(self, reference_data: pd.DataFrame):
        pass

    def _estimate(self, chunks: List[Chunk]) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    'key': chunk.key,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'start_date': chunk.start_datetime,
                    'end_date': chunk.end_datetime,
                    'partition': 'analysis' if chunk.is_transition else chunk.partition,
                    'estimated_performance': _calculate_cme_for_chunk(chunk),
                }
                for chunk in chunks
            ]
        )


def _calculate_cme_for_chunk(chunk: Chunk) -> float:
    return np.random.randn()
