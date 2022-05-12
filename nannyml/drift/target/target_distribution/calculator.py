#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module for target distribution monitoring."""
from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from nannyml.chunk import Chunker, CountBasedChunker, DefaultChunker, PeriodBasedChunker, SizeBasedChunker
from nannyml.drift.target.target_distribution.result import TargetDistributionResult
from nannyml.exceptions import InvalidArgumentsException
from nannyml.metadata.base import (
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

        Examples
        --------
        >>> import nannyml as nml
        >>> ref_df, ana_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>> metadata = nml.extract_metadata(ref_df, model_type=nml.ModelType.CLASSIFICATION_BINARY)
        >>> # Create a calculator that will chunk by week
        >>> target_distribution_calc = nml.TargetDistributionCalculator(model_metadata=metadata, chunk_period='W')
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

    def fit(self, reference_data: pd.DataFrame) -> TargetDistributionCalculator:
        """Fits the calculator to reference data.

        During fitting the reference target data is validated and stored for later use.

        Examples
        --------
        >>> import nannyml as nml
        >>> ref_df, ana_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>> metadata = nml.extract_metadata(ref_df, model_type=nml.ModelType.CLASSIFICATION_BINARY)
        >>> target_distribution_calc = nml.TargetDistributionCalculator(model_metadata=metadata, chunk_period='W')
        >>> # fit the calculator on reference data
        >>> target_distribution_calc.fit(ref_df)
        """
        if reference_data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        if self.metadata.target_column_name not in reference_data.columns:
            raise InvalidArgumentsException(
                f"data does not contain target data column '{self.metadata.target_column_name}'."
            )

        self._reference_targets = preprocess(data=reference_data, metadata=self.metadata, reference=True)[
            NML_METADATA_TARGET_COLUMN_NAME
        ]

        return self

    def calculate(self, data: pd.DataFrame):
        """Calculates the target distribution of a binary classifier.

        Requires fitting the calculator on reference data first.

        Parameters
        ----------
        data: pd.DataFrame
            Data for the model, i.e. model inputs, predictions and targets.

        Examples
        --------
        >>> import nannyml as nml
        >>> ref_df, ana_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>> metadata = nml.extract_metadata(ref_df, model_type=nml.ModelType.CLASSIFICATION_BINARY)
        >>> target_distribution_calc = nml.TargetDistributionCalculator(model_metadata=metadata, chunk_period='W')
        >>> target_distribution_calc.fit(ref_df)
        >>> # calculate target distribution
        >>> target_distribution = target_distribution_calc.calculate(ana_df)
        """
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        if self.metadata.target_column_name not in data.columns:
            raise InvalidArgumentsException(
                f"data does not contain target data column '{self.metadata.target_column_name}'."
            )

        # Preprocess data
        data = preprocess(data=data, metadata=self.metadata)

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

    is_binary_targets = data[NML_METADATA_TARGET_COLUMN_NAME].nunique() > 2
    if is_binary_targets:
        warnings.warn(
            f"the target column contains {data[NML_METADATA_TARGET_COLUMN_NAME].nunique()} unique values. "
            "NannyML cannot provide a value for 'metric_target_drift' "
            "when there are more than 2 unique values. "
            "All 'metric_target_drift' values will be set to np.NAN"
        )

    is_string_targets = (
        data[NML_METADATA_TARGET_COLUMN_NAME].dtype == 'object'
        or data[NML_METADATA_TARGET_COLUMN_NAME].dtype == 'string'
    )
    if is_string_targets:
        warnings.warn(
            "the target column contains non-numerical values. NannyML cannot provide a value for "
            "'metric_target_drift'."
            "All 'metric_target_drift' values will be set to np.NAN"
        )

    return {
        'metric_target_drift': targets.mean() if not (is_binary_targets or is_string_targets) else np.NAN,
        'statistical_target_drift': statistic,
        'p_value': p_value,
        'thresholds': _ALERT_THRESHOLD_P_VALUE,
        'alert': (p_value < _ALERT_THRESHOLD_P_VALUE) and is_analysis,
        'significant': p_value < _ALERT_THRESHOLD_P_VALUE,
    }
