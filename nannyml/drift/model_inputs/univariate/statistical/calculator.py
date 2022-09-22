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
from nannyml.drift.model_inputs.univariate.statistical.results import Result
from nannyml.exceptions import InvalidArgumentsException

ALERT_THRESHOLD_P_VALUE = 0.05


class UnivariateStatisticalDriftCalculator(AbstractCalculator):
    """Calculates drift for individual features using statistical tests."""

    def __init__(
        self,
        feature_column_names: List[str],
        timestamp_column_name: str = None,
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
        timestamp_column_name: str, default=None
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
        >>> from IPython.display import display
        >>> reference_df = nml.load_synthetic_binary_classification_dataset()[0]
        >>> analysis_df = nml.load_synthetic_binary_classification_dataset()[1]
        >>> display(reference_df.head())
        >>> feature_column_names = [
        ...     col for col in reference_df.columns if col not in [
        ...     'timestamp', 'y_pred_proba', 'period', 'y_pred', 'work_home_actual', 'identifier'
        >>> ]]
        >>> calc = nml.UnivariateStatisticalDriftCalculator(
        ...     feature_column_names=feature_column_names,
        ...     timestamp_column_name='timestamp'
        >>> )
        >>> calc.fit(reference_df)
        >>> results = calc.calculate(analysis_df)
        >>> display(results.data.iloc[:, :9])
        >>> display(calc.previous_reference_results.iloc[:, :9])
        >>> for feature in calc.feature_column_names:
        ...     drift_fig = results.plot(
        ...         kind='feature_drift',
        ...         feature_column_name=feature,
        ...         plot_reference=True
        ...     )
        ...     drift_fig.show()
        >>> for cont_feat in calc.continuous_column_names:
        ...     figure = results.plot(
        ...         kind='feature_distribution',
        ...         feature_column_name=cont_feat,
        ...         plot_reference=True
        ...     )
        ...     figure.show()
        >>> for cat_feat in calc.categorical_column_names:
        ...     figure = results.plot(
        ...         kind='feature_distribution',
        ...         feature_column_name=cat_feat,
        ...         plot_reference=True)
        ...     figure.show()
        >>> ranker = nml.Ranker.by('alert_count')
        >>> ranked_features = ranker.rank(results, only_drifting = False)
        >>> display(ranked_features)
        """
        super(UnivariateStatisticalDriftCalculator, self).__init__(
            chunk_size, chunk_number, chunk_period, chunker, timestamp_column_name
        )

        self.feature_column_names = feature_column_names
        self.continuous_column_names: List[str] = []
        self.categorical_column_names: List[str] = []

        # required for distribution plots
        self.previous_reference_data: Optional[pd.DataFrame] = None
        self.previous_reference_results: Optional[pd.DataFrame] = None
        self.previous_analysis_data: Optional[pd.DataFrame] = None

    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> UnivariateStatisticalDriftCalculator:
        """Fits the drift calculator using a set of reference data."""
        if reference_data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing(self.feature_column_names, reference_data)

        # Reference stability
        self._reference_stability = 0  # TODO: Jakub

        self.previous_reference_data = reference_data.copy()
        self.previous_reference_results = self._calculate(self.previous_reference_data).data

        return self

    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> Result:
        """Calculates the data reconstruction drift for a given data set."""
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing(self.feature_column_names, data)

        self.continuous_column_names, self.categorical_column_names = _split_features_by_type(
            data, self.feature_column_names
        )

        chunks = self.chunker.split(data)

        chunk_drifts = []
        # Calculate chunk-wise drift statistics.
        # Append all into resulting DataFrame indexed by chunk key.
        for chunk in chunks:
            chunk_drift: Dict[str, Any] = {
                'key': chunk.key,
                'chunk_index': chunk.chunk_index,
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

        from nannyml.drift.model_inputs.univariate.statistical.results import Result

        return Result(results_data=res, calculator=self)
