#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Statistical drift calculation using `Kolmogorov-Smirnov` and `chi2-contingency` tests."""
import itertools
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from nannyml.chunk import Chunk
from nannyml.drift._base import BaseDriftCalculator, ChunkerPreset
from nannyml.metadata import ModelMetadata


class StatisticalDriftCalculator(BaseDriftCalculator):
    """A drift calculator that relies on statistics to detect drift."""

    def _calculate_drift(
        self, reference_data: pd.DataFrame, chunks: List[Chunk], model_metadata: ModelMetadata
    ) -> pd.DataFrame:
        # Get lists of categorical <-> categorical features
        categorical_column_names = [f.column_name for f in model_metadata.categorical_features]
        continuous_column_names = [f.column_name for f in model_metadata.continuous_features]

        res = pd.DataFrame()
        # Calculate chunk-wise drift statistics.
        # Append all into resulting DataFrame indexed by chunk key.
        for chunk in chunks:
            chunk_drift: Dict[str, Any] = {'chunk': chunk.key}

            present_categorical_column_names = list(set(chunk.data.columns) & set(categorical_column_names))
            for column in present_categorical_column_names:
                statistic, p_value, _, _ = chi2_contingency(
                    pd.concat([reference_data[column].value_counts(), chunk.data[column].value_counts()], axis=1)
                )
                chunk_drift[f'{column}_statistic'] = [statistic]
                chunk_drift[f'{column}_p_value'] = [np.round(p_value, decimals=3)]

            present_continuous_column_names = list(set(chunk.data.columns) & set(continuous_column_names))
            for column in present_continuous_column_names:
                statistic, p_value = ks_2samp(reference_data[column], chunk.data[column])
                chunk_drift[f'{column}_statistic'] = [statistic]
                chunk_drift[f'{column}_p_value'] = [np.round(p_value, decimals=3)]

            res = res.append(pd.DataFrame(chunk_drift))

        res = res.reset_index(drop=True)
        return res


def calculate_statistical_drift(
    reference_data: pd.DataFrame,
    analysis_data: pd.DataFrame,
    model_metadata: ModelMetadata,
    chunk_by: Union[str, ChunkerPreset] = 'size_1000',
) -> pd.DataFrame:
    """Calculates drift using statistical testing.

    This function constructs a StatisticalDriftCalculator and subsequently uses it to calculate drift on a DataFrame
    of analysis data against a reference DataFrame.

    """
    calculator = StatisticalDriftCalculator()
    return calculator.calculate(reference_data, analysis_data, model_metadata, chunk_by=chunk_by)


def _map_by_index(reference_chunks: List[Chunk], analysis_chunks: List[Chunk]) -> Dict[Chunk, Chunk]:
    return {l1: l2 for l1, l2 in itertools.zip_longest(reference_chunks, analysis_chunks, fillvalue=None)}
