#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Calculates drift for individual features using the `Kolmogorov-Smirnov` and `chi2-contingency` statistical tests."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from nannyml.base import AbstractCalculator, _list_missing
from nannyml.chunk import Chunker
from nannyml.drift.model_inputs.univariate.distance.metrics import Metric, MetricFactory
from nannyml.drift.model_inputs.univariate.distance.results import Result
from nannyml.exceptions import InvalidArgumentsException

ALERT_THRESHOLD_DISTANCE = 0.1


class DistanceDriftCalculator(AbstractCalculator):
    """Calculates drift for individual features using statistical tests."""

    def __init__(
        self,
        feature_column_names: List[str],
        metrics: List[str],
        timestamp_column_name: Optional[str],
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
        metrics: List[str]
            A list of metrics to calculate. Must be one of: `jensen_shannon`.
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
        >>> reference, analysis, _ = nml.load_synthetic_car_price_dataset()
        >>> calc = nml.DistanceDriftCalculator(
        ...     timestamp_column_name='timestamp',
        ...     metrics=['jensen_shannon'],
        ...     feature_column_names=[col for col in reference.columns if col not in ['timestamp', 'y_pred', 'y_true']]
        ... ).fit(reference)
        >>> res = calc.calculate(analysis)
        >>> for feature in calc.feature_column_names:
        ...     for metric in calc.metrics:
        ...         res.plot(kind='feature_distribution', feature_column_name=feature, metric=metric).show()
        """
        super(DistanceDriftCalculator, self).__init__(
            chunk_size, chunk_number, chunk_period, chunker, timestamp_column_name
        )

        self.feature_column_names = feature_column_names

        self.metrics: List[Metric] = [MetricFactory.create(m, {'calculator': self}) for m in metrics]  # type: ignore

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
                for metric in self.metrics:
                    distance = metric.calculate(chunk.data, feature)
                    chunk_drift[f'{feature}_{metric.column_name}'] = distance
                    chunk_drift[f'{feature}_{metric.column_name}_alert'] = (
                        metric.lower_threshold is not None and distance < metric.lower_threshold
                    ) or (metric.upper_threshold is not None and distance > metric.upper_threshold)
                    chunk_drift[f'{feature}_{metric.column_name}_upper_threshold'] = metric.upper_threshold
                    chunk_drift[f'{feature}_{metric.column_name}_lower_threshold'] = metric.lower_threshold

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
