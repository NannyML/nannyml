#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Calculates drift for model targets and target distributions using statistical tests."""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from nannyml._typing import ProblemType
from nannyml.base import AbstractCalculator
from nannyml.chunk import Chunker
from nannyml.drift.target.target_distribution.result import Result
from nannyml.exceptions import CalculatorNotFittedException, InvalidArgumentsException

_ALERT_THRESHOLD_P_VALUE = 0.05


class TargetDistributionCalculator(AbstractCalculator):
    """Calculates drift for model targets and target distributions using statistical tests."""

    def __init__(
        self,
        y_true: str,
        problem_type: Union[str, ProblemType],
        timestamp_column_name: str = None,
        chunk_size: int = None,
        chunk_number: int = None,
        chunk_period: str = None,
        chunker: Chunker = None,
    ):
        """creates a new TargetDistributionCalculator.

        Parameters
        ----------
        y_true: str
            The name of the column containing your model target values.
        timestamp_column_name: str, default=None
            The name of the column containing the timestamp of the model prediction.
        chunk_size: int, default=None
            Splits the data into chunks containing `chunks_size` observations.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunk_number: int, default=None
            Splits the data into `chunk_number` pieces.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunk_period: str, default=None
            Splits the data according to the given period.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunker : Chunker, default=None
            The `Chunker` used to split the data sets into a lists of chunks.

        Examples
        --------
        >>> import nannyml as nml
        >>> from IPython.display import display
        >>> reference_df = nml.load_synthetic_binary_classification_dataset()[0]
        >>> analysis_df = nml.load_synthetic_binary_classification_dataset()[1]
        >>> analysis_target_df = nml.load_synthetic_binary_classification_dataset()[2]
        >>> analysis_df = analysis_df.merge(analysis_target_df, on='identifier')
        >>> display(reference_df.head(3))
        >>> calc = nml.TargetDistributionCalculator(
        ...     y_true='work_home_actual',
        ...     timestamp_column_name='timestamp',
        ...     problem_type='classification_binary'
        >>> )
        >>> calc.fit(reference_df)
        >>> results = calc.calculate(analysis_df)
        >>> display(results.data.head(3))
        >>> target_drift_fig = results.plot(kind='target_drift', plot_reference=True)
        >>> target_drift_fig.show()
        >>> target_distribution_fig = results.plot(kind='target_distribution', plot_reference=True)
        >>> target_distribution_fig.show()
        """
        super().__init__(chunk_size, chunk_number, chunk_period, chunker, timestamp_column_name)

        self.y_true = y_true

        if isinstance(problem_type, str):
            problem_type = ProblemType.parse(problem_type)
        self.problem_type: ProblemType = problem_type  # type: ignore

        self.previous_reference_results: Optional[pd.DataFrame] = None
        self.previous_reference_data: Optional[pd.DataFrame] = None
        self.previous_analysis_data: Optional[pd.DataFrame] = None

    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> TargetDistributionCalculator:
        """Fits the calculator to reference data."""
        if reference_data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        reference_data = reference_data.copy()

        if self.y_true not in reference_data.columns:
            raise InvalidArgumentsException(
                f"target data column '{self.y_true}' is not in data columns: {reference_data.columns}."
            )

        # Reference stability
        self._reference_stability = 0  # TODO: Jakub

        self.previous_reference_data = reference_data
        self.previous_reference_results = self._calculate(reference_data).data

        return self

    def _calculate(self, data: pd.DataFrame, *args, **kwargs):
        """Calculates the target distribution of a binary classifier."""
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        data = data.copy()

        if self.y_true not in data.columns:
            raise InvalidArgumentsException(
                f"target data column '{self.y_true}' not found in data columns: {data.columns}."
            )

        data['NML_TARGET_INCOMPLETE'] = data[self.y_true].isna().astype(np.int16)

        # Generate chunks
        # features_and_metadata = NML_METADATA_COLUMNS + ['NML_TARGET_INCOMPLETE']
        chunks = self.chunker.split(
            data,
            columns=[self.y_true, 'NML_TARGET_INCOMPLETE'],
        )

        # Construct result frame
        if self.previous_reference_data is None:
            raise CalculatorNotFittedException("no reference data known. Did you fit the calculator first?")
        res = pd.DataFrame.from_records(
            [
                {
                    'key': chunk.key,
                    'chunk_index': chunk.chunk_index,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'start_date': chunk.start_datetime,
                    'end_date': chunk.end_datetime,
                    'period': 'analysis' if chunk.is_transition else chunk.period,
                    'targets_missing_rate': (
                        chunk.data['NML_TARGET_INCOMPLETE'].sum() / chunk.data['NML_TARGET_INCOMPLETE'].count()
                    ),
                    **self._calculate_target_drift_for_chunk(
                        self.previous_reference_data[self.y_true], chunk.data[self.y_true]
                    ),
                }
                for chunk in chunks
            ]
        )

        self.previous_analysis_data = data.copy()

        return Result(results_data=res, calculator=self)

    def _calculate_target_drift_for_chunk(self, reference_targets: pd.Series, analysis_targets: pd.Series) -> Dict:
        if self.problem_type in [ProblemType.CLASSIFICATION_BINARY, ProblemType.CLASSIFICATION_MULTICLASS]:
            return _calculate_categorical_target_drift_for_chunk(reference_targets, analysis_targets)
        elif self.problem_type in [ProblemType.REGRESSION]:
            return _calculate_continuous_target_drift_for_chunk(reference_targets, analysis_targets)
        else:
            raise InvalidArgumentsException(
                f"target drift calculation is not support for '{ProblemType.value}' problems"
            )


def _calculate_categorical_target_drift_for_chunk(reference_targets: pd.Series, targets: pd.Series) -> Dict:
    statistic, p_value, _, _ = chi2_contingency(
        pd.concat([reference_targets.value_counts(), targets.value_counts()], axis=1).fillna(0)
    )

    is_non_binary_targets = targets.nunique() > 2
    if is_non_binary_targets:
        warnings.warn(
            f"the target column contains {targets.nunique()} unique values. "
            "NannyML cannot provide a value for 'metric_target_drift' "
            "when there are more than 2 unique values. "
            "All 'metric_target_drift' values will be set to np.NAN"
        )

    is_string_targets = targets.dtype in ['object', 'string']
    if is_string_targets:
        warnings.warn(
            "the target column contains non-numerical values. NannyML cannot provide a value for "
            "'metric_target_drift'."
            "All 'metric_target_drift' values will be set to np.NAN"
        )

    return {
        'metric_target_drift': targets.mean() if not (is_non_binary_targets or is_string_targets) else np.NAN,
        'statistical_target_drift': statistic,
        'p_value': p_value,
        'thresholds': _ALERT_THRESHOLD_P_VALUE,
        'alert': p_value < _ALERT_THRESHOLD_P_VALUE,
        'significant': p_value < _ALERT_THRESHOLD_P_VALUE,
    }


def _calculate_continuous_target_drift_for_chunk(reference_targets: pd.Series, targets: pd.Series) -> Dict:
    statistic, p_value = ks_2samp(reference_targets, targets)
    return {
        'metric_target_drift': targets.mean(),
        'statistical_target_drift': statistic,
        'p_value': p_value,
        'thresholds': _ALERT_THRESHOLD_P_VALUE,
        'alert': p_value < _ALERT_THRESHOLD_P_VALUE,
        'significant': p_value < _ALERT_THRESHOLD_P_VALUE,
    }
