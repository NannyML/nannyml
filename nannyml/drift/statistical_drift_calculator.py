#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Statistical drift calculation using `Kolmogorov-Smirnov` and `chi2-contingency` tests."""
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from nannyml.chunk import Chunk
from nannyml.drift._base import BaseDriftCalculator
from nannyml.metadata import ModelMetadata


class StatisticalDriftCalculator(BaseDriftCalculator):
    """A drift calculator that relies on statistics to detect drift."""

    def _calculate_drift(
        self,
        reference_data: pd.DataFrame,
        chunks: List[Chunk],
        model_metadata: ModelMetadata,
        selected_features: List[str],
    ) -> pd.DataFrame:
        # Get lists of categorical <-> categorical features
        categorical_column_names = [f.column_name for f in model_metadata.categorical_features]
        continuous_column_names = [f.column_name for f in model_metadata.continuous_features]

        res = pd.DataFrame()
        # Calculate chunk-wise drift statistics.
        # Append all into resulting DataFrame indexed by chunk key.
        for chunk in chunks:
            chunk_drift: Dict[str, Any] = {
                'key': chunk.key,
                'start_index': chunk.start_index,
                'end_index': chunk.end_index,
                'start_date': chunk.start_datetime,
                'end_date': chunk.end_datetime,
                'partition': 'analysis' if chunk.is_transition else chunk.partition,
            }

            present_categorical_column_names = list(set(chunk.data.columns) & set(categorical_column_names))
            for column in present_categorical_column_names:
                statistic, p_value, _, _ = chi2_contingency(
                    pd.concat([reference_data[column].value_counts(), chunk.data[column].value_counts()], axis=1)
                )
                chunk_drift[f'{column}_chi2'] = [statistic]
                chunk_drift[f'{column}_p_value'] = [np.round(p_value, decimals=3)]

            present_continuous_column_names = list(set(chunk.data.columns) & set(continuous_column_names))
            for column in present_continuous_column_names:
                statistic, p_value = ks_2samp(reference_data[column], chunk.data[column])
                chunk_drift[f'{column}_dstat'] = [statistic]
                chunk_drift[f'{column}_p_value'] = [np.round(p_value, decimals=3)]

            res = res.append(pd.DataFrame(chunk_drift))

        res = res.reset_index(drop=True)
        return res


def calculate_statistical_drift(
    reference_data: pd.DataFrame,
    analysis_data: pd.DataFrame,
    model_metadata: ModelMetadata,
    chunk_size: int = None,
    chunk_number: int = None,
    chunk_period: str = None,
) -> pd.DataFrame:
    """Calculates drift using statistical testing.

    This function constructs a StatisticalDriftCalculator and subsequently uses it to calculate drift on a DataFrame
    of analysis data against a reference DataFrame.

    """
    calculator = StatisticalDriftCalculator()
    return calculator.calculate(reference_data, analysis_data, model_metadata, chunk_size, chunk_number, chunk_period)
