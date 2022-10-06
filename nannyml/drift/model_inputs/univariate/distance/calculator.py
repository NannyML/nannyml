#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Calculates drift for individual features using the `Kolmogorov-Smirnov` and `chi2-contingency` statistical tests."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial import distance

from nannyml.base import AbstractCalculator, _list_missing, _column_is_continuous
from nannyml.chunk import Chunker
from nannyml.drift.model_inputs.univariate.distance.results import UnivariateDistanceDriftCalculatorResult
from nannyml.exceptions import InvalidArgumentsException

ALERT_THRESHOLD_DISTANCE = 0.1


class UnivariateDistanceDriftCalculator(AbstractCalculator):
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
        super(UnivariateDistanceDriftCalculator, self).__init__(chunk_size, chunk_number, chunk_period, chunker)

        self.feature_column_names = feature_column_names
        self.continuous_column_names: List[str] = []
        self.categorical_column_names: List[str] = []

        self.timestamp_column_name = timestamp_column_name

        # required for distribution plots
        self.previous_reference_data: Optional[pd.DataFrame] = None
        self.previous_reference_results: Optional[pd.DataFrame] = None
        self.previous_analysis_data: Optional[pd.DataFrame] = None

    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> UnivariateDistanceDriftCalculator:
        """Fits the drift calculator using a set of reference data."""
        if reference_data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing(self.feature_column_names, reference_data)

        self.previous_reference_data = reference_data.copy()
        self.previous_reference_results = self._calculate(self.previous_reference_data).data

        return self

    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> UnivariateDistanceDriftCalculatorResult:
        """Calculates the jenson-shannon divergence for a given data set."""
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing(self.feature_column_names, data)

        # self.continuous_column_names, self.categorical_column_names = _split_features_by_type(
        #     data, self.feature_column_names
        # )

        chunks = self.chunker.split(data)

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

            for feature in self.feature_column_names:
                ref_binned_data, ana_binned_data = get_binned_data(self.previous_reference_data[feature], chunk.data[feature])
                dis = distance.jensenshannon(ref_binned_data, ana_binned_data)
                pd.concat(
                    [
                        self.previous_reference_data[feature].value_counts(),  # type: ignore
                        chunk.data[feature].value_counts(),
                    ],
                    axis=1,
                ).fillna(0)
                chunk_drift[f'{feature}_js'] = dis
                chunk_drift[f'{feature}_alert'] = dis > ALERT_THRESHOLD_DISTANCE
                chunk_drift[f'{feature}_threshold'] = ALERT_THRESHOLD_DISTANCE

            chunk_drifts.append(chunk_drift)

        res = pd.DataFrame.from_records(chunk_drifts)
        res = res.reset_index(drop=True)

        self.previous_analysis_data = data

        from nannyml.drift.model_inputs.univariate.statistical.results import UnivariateStatisticalDriftCalculatorResult

        return UnivariateStatisticalDriftCalculatorResult(results_data=res, calculator=self)


def get_binned_data(reference_feature: pd.Series, analysis_feature: pd.Series):
    """Split variable into n buckets based on reference quantiles
    Args:
        reference_feature: reference data
        analysis_feature: analysis data
    Returns:
        ref_binned_pdf: probability estimate in each bucket for reference
        curr_binned_pdf: probability estimate in each bucket for reference
    """
    n_vals = reference_feature.nunique()
    if _column_is_continuous(reference_feature) == "num" and n_vals > 20:
        bins = np.histogram_bin_edges(list(reference_feature) + list(analysis_feature), bins="sturges")
        refq = pd.cut(reference_feature, bins=bins)
        anaq = pd.cut(analysis_feature, bins=bins)
        ref_binned_pdf = list(refq.value_counts(sort=False)/len(reference_feature))
        ana_binned_pdf = list(anaq.value_counts(sort=False)/len(analysis_feature))

    else:
        keys = list((set(reference_feature.unique()) | set(analysis_feature.unique())) - {np.nan})
        ref_binned_pdf = [(reference_feature == i).sum()/len(reference_feature) for i in keys]
        ana_binned_pdf = [(analysis_feature == i).sum()/len(analysis_feature) for i in keys]

    return ref_binned_pdf, ana_binned_pdf
