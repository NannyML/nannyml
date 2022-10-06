#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Calculates drift for individual features using the `Kolmogorov-Smirnov` and `chi2-contingency` statistical tests."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial import distance

from nannyml.base import AbstractCalculator, _column_is_continuous, _list_missing
from nannyml.chunk import Chunker
from nannyml.drift.model_inputs.univariate.distance.results import Result
from nannyml.exceptions import InvalidArgumentsException

ALERT_THRESHOLD_DISTANCE = 0.1


class DistanceDriftCalculator(AbstractCalculator):
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
        # TODO provide example
        """
        super(DistanceDriftCalculator, self).__init__(
            chunk_size, chunk_number, chunk_period, chunker, timestamp_column_name
        )

        self.feature_column_names = feature_column_names

        # required for distribution plots
        self.previous_reference_results: Optional[pd.DataFrame] = None
        self.previous_analysis_data: Optional[pd.DataFrame] = None

        self.result: Optional[Result] = None

    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> DistanceDriftCalculator:
        """Fits the drift calculator using a set of reference data."""
        if reference_data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing(self.feature_column_names, reference_data)

        self.previous_reference_data = reference_data.copy()
        self.result = self._calculate(self.previous_reference_data)
        self.result.data['period'] = 'reference'

        return self

    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> Result:
        """Calculates the jenson-shannon divergence for a given data set."""
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing(self.feature_column_names, data)

        chunks = self.chunker.split(data)

        chunk_drifts = []
        for chunk in chunks:
            chunk_drift: Dict[str, Any] = {
                'key': chunk.key,
                'chunk_index': chunk.chunk_index,
                'start_index': chunk.start_index,
                'end_index': chunk.end_index,
                'start_date': chunk.start_datetime,
                'end_date': chunk.end_datetime,
            }

            for feature in self.feature_column_names:
                ref_binned_data, ana_binned_data = get_binned_data(
                    self.previous_reference_data[feature], chunk.data[feature]
                )
                dis = distance.jensenshannon(ref_binned_data, ana_binned_data)
                pd.concat(
                    [
                        self.previous_reference_data[feature].value_counts(),  # type: ignore
                        chunk.data[feature].value_counts(),
                    ],
                    axis=1,
                ).fillna(0)
                chunk_drift[f'{feature}_jensen_shannon'] = dis
                chunk_drift[f'{feature}_alert'] = dis > ALERT_THRESHOLD_DISTANCE
                chunk_drift[f'{feature}_threshold'] = ALERT_THRESHOLD_DISTANCE

            chunk_drifts.append(chunk_drift)

        res = pd.DataFrame.from_records(chunk_drifts)
        res = res.reset_index(drop=True)
        res['period'] = 'analysis'

        self.previous_analysis_data = data

        if self.result is None:
            self.result = Result(results_data=res, calculator=self)
        else:
            self.result.data = pd.concat([self.result.data, res]).reset_index(drop=True)

        return self.result


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
    if _column_is_continuous(reference_feature) and n_vals > 20:
        bins = np.histogram_bin_edges(list(reference_feature) + list(analysis_feature), bins="sturges")
        refq = pd.cut(reference_feature, bins=bins)
        anaq = pd.cut(analysis_feature, bins=bins)
        ref_binned_pdf = list(refq.value_counts(sort=False) / len(reference_feature))
        ana_binned_pdf = list(anaq.value_counts(sort=False) / len(analysis_feature))

    else:
        keys = list((set(reference_feature.unique()) | set(analysis_feature.unique())) - {np.nan})
        ref_binned_pdf = [(reference_feature == i).sum() / len(reference_feature) for i in keys]
        ana_binned_pdf = [(analysis_feature == i).sum() / len(analysis_feature) for i in keys]

    return ref_binned_pdf, ana_binned_pdf
