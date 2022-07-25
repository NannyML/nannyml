#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Calculates drift for individual features using the `Kolmogorov-Smirnov` and `chi2-contingency` statistical tests."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from nannyml.base import AbstractCalculator, _list_missing, _split_features_by_type
from nannyml.chunk import Chunker
from nannyml.drift.model_inputs.univariate.statistical.results import UnivariateStatisticalDriftCalculatorResult
from nannyml.exceptions import InvalidArgumentsException

ALERT_THRESHOLD_P_VALUE = 0.05


class UnivariateStatisticalDriftCalculator(AbstractCalculator):
    """Calculates drift for individual features using statistical tests."""

    def __init__(
        self,
        feature_column_names: List[str],
        timestamp_column_name: str,
        chunk_size: int = None,
        chunk_number: int = None,
        chunk_period: str = None,
        chunker: Chunker = None,
    ):
        """Creates a new UnivariateStatisticalDriftCalculator instance.

        Parameters
        ----------
        feature_column_names: List[str]
            A list containing the names of features in the provided data set.
            A drift score will be calculated for each entry in this list.
        timestamp_column_name: str
            The name of the column containing the timestamp of the model prediction.
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
        >>>
        >>> reference_df, analysis_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>>
        >>> feature_column_names = [col for col in reference_df.columns
        >>>                         if col not in ['y_pred', 'y_pred_proba', 'work_home_actual', 'timestamp']]
        >>> calc = nml.UnivariateStatisticalDriftCalculator(
        >>>     feature_column_names=feature_column_names,
        >>>     timestamp_column_name='timestamp'
        >>> )
        >>> calc.fit(reference_df)
        >>> results = calc.calculate(analysis_df)
        >>> print(results.data)  # check the numbers
                     key  start_index  ...  identifier_alert identifier_threshold
        0       [0:4999]            0  ...              True                 0.05
        1    [5000:9999]         5000  ...              True                 0.05
        2  [10000:14999]        10000  ...              True                 0.05
        3  [15000:19999]        15000  ...              True                 0.05
        4  [20000:24999]        20000  ...              True                 0.05
        5  [25000:29999]        25000  ...              True                 0.05
        6  [30000:34999]        30000  ...              True                 0.05
        7  [35000:39999]        35000  ...              True                 0.05
        8  [40000:44999]        40000  ...              True                 0.05
        9  [45000:49999]        45000  ...              True                 0.05
        >>> fig = results.plot(kind='feature_drift', plot_reference=True, feature_column_name='distance_from_office')
        >>> fig.show()
        """
        super(UnivariateStatisticalDriftCalculator, self).__init__(chunk_size, chunk_number, chunk_period, chunker)

        self.feature_column_names = feature_column_names
        self.continuous_column_names: List[str] = []
        self.categorical_column_names: List[str] = []

        self.timestamp_column_name = timestamp_column_name

        # required for distribution plots
        self.previous_reference_data: Optional[pd.DataFrame] = None
        self.previous_reference_results: Optional[pd.DataFrame] = None
        self.previous_analysis_data: Optional[pd.DataFrame] = None

    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> UnivariateStatisticalDriftCalculator:
        """Fits the drift calculator using a set of reference data."""
        if reference_data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing(self.feature_column_names, reference_data)

        self.previous_reference_data = reference_data.copy()
        self.previous_reference_results = self._calculate(self.previous_reference_data).data

        return self

    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> UnivariateStatisticalDriftCalculatorResult:
        """Calculates the data reconstruction drift for a given data set."""
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing(self.feature_column_names, data)

        self.continuous_column_names, self.categorical_column_names = _split_features_by_type(
            data, self.feature_column_names
        )

        # features_and_metadata = NML_METADATA_COLUMNS + self.selected_features
        chunks = self.chunker.split(
            data,
            self.timestamp_column_name,
            # columns=self.feature_column_names + [NML_METADATA_PERIOD_COLUMN_NAME],
            minimum_chunk_size=500,
        )

        chunk_drifts = []
        # Calculate chunk-wise drift statistics.
        # Append all into resulting DataFrame indexed by chunk key.
        for chunk in chunks:
            chunk_drift: Dict[str, Any] = {
                'key': chunk.key,
                'start_index': chunk.start_index,
                'end_index': chunk.end_index,
                'start_date': chunk.start_datetime,
                'end_date': chunk.end_datetime,
            }

            for column in self.categorical_column_names:
                statistic, p_value, _, _ = chi2_contingency(
                    pd.concat(
                        [
                            self.previous_reference_data[column].value_counts(),  # type: ignore
                            chunk.data[column].value_counts(),
                        ],
                        axis=1,
                    ).fillna(0)
                )
                chunk_drift[f'{column}_chi2'] = statistic
                chunk_drift[f'{column}_p_value'] = np.round(p_value, decimals=3)
                chunk_drift[f'{column}_alert'] = p_value < ALERT_THRESHOLD_P_VALUE
                chunk_drift[f'{column}_threshold'] = ALERT_THRESHOLD_P_VALUE

            for column in self.continuous_column_names:
                statistic, p_value = ks_2samp(self.previous_reference_data[column], chunk.data[column])  # type: ignore
                chunk_drift[f'{column}_dstat'] = statistic
                chunk_drift[f'{column}_p_value'] = np.round(p_value, decimals=3)
                chunk_drift[f'{column}_alert'] = p_value < ALERT_THRESHOLD_P_VALUE
                chunk_drift[f'{column}_threshold'] = ALERT_THRESHOLD_P_VALUE

            chunk_drifts.append(chunk_drift)

        res = pd.DataFrame.from_records(chunk_drifts)
        res = res.reset_index(drop=True)

        self.previous_analysis_data = data

        from nannyml.drift.model_inputs.univariate.statistical.results import UnivariateStatisticalDriftCalculatorResult

        return UnivariateStatisticalDriftCalculatorResult(results_data=res, calculator=self)
