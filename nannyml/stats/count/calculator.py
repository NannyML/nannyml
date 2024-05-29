#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

"""Simple Statistics Average Calculator."""

from typing import Any, Dict, Optional

import pandas as pd
from pandas import MultiIndex

from nannyml.base import AbstractCalculator
from nannyml.chunk import Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.stats.count.result import Result
from nannyml.thresholds import StandardDeviationThreshold, Threshold, calculate_threshold_values
from nannyml.usage_logging import UsageEvent, log_usage


class SummaryStatsRowCountCalculator(AbstractCalculator):
    """SummaryStatsRowCountCalculator implementation."""

    def __init__(
        self,
        timestamp_column_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_number: Optional[int] = None,
        chunk_period: Optional[str] = None,
        chunker: Optional[Chunker] = None,
        threshold: Threshold = StandardDeviationThreshold(),
    ):
        """Creates a new SummaryStatsRowCountCalculator instance.

        Parameters
        ----------
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
        threshold: Appropriate `Threshold` subclass.
            Defines alert thresholds strategy.
            Defaults to StandardDeviationThreshold()


        Examples
        --------
        >>> import nannyml as nml
        >>> reference, analysis, _ = nml.load_synthetic_car_price_dataset()
        >>> calc = nml.SummaryStatsRowCountCalculator(
        ...     timestamp_column_name='timestamp',
        ... ).fit(reference)
        >>> res = calc.calculate(analysis)
        >>> res.plot().show()
        """
        super(SummaryStatsRowCountCalculator, self).__init__(
            chunk_size, chunk_number, chunk_period, chunker, timestamp_column_name
        )

        self.result: Optional[Result] = None
        # No sampling error
        # threshold strategy is the same across all columns
        self.threshold = threshold
        self._upper_alert_threshold: Optional[float] = 0
        self._lower_alert_threshold: Optional[float] = 0

        self.lower_threshold_value_limit: Optional[float] = 0
        self.upper_threshold_value_limit: Optional[float] = None
        self.simple_stats_metric = 'rows_count'

    def _calculate_count_value_stats(self, data: pd.DataFrame):
        # count vs shape have slightly different behaviors!
        # count ignores rows with missing values, infringing a bit on missing values calc so shape
        return data.shape[0]

    @log_usage(UsageEvent.STATS_COUNT_FIT)
    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs):
        """Fits the drift calculator to a set of reference data."""
        if reference_data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        self.result = self._calculate(data=reference_data)
        self.result.data[('chunk', 'period')] = 'reference'

        return self

    @log_usage(UsageEvent.STATS_COUNT_RUN)
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> Result:
        """Calculates methods for both categorical and continuous columns."""
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        chunks = self.chunker.split(data)

        rows = []
        for chunk in chunks:
            row = {
                'key': chunk.key,
                'chunk_index': chunk.chunk_index,
                'start_index': chunk.start_index,
                'end_index': chunk.end_index,
                'start_datetime': chunk.start_datetime,
                'end_datetime': chunk.end_datetime,
                'period': 'analysis',
            }

            for k, v in self._calculate_for_df(chunk.data).items():
                row[f'{self.simple_stats_metric}_{k}'] = v

            rows.append(row)

        result_index = _create_multilevel_index(
            column0=self.simple_stats_metric,
        )
        res = pd.DataFrame(rows)
        res.columns = result_index
        res = res.reset_index(drop=True)

        if self.result is None:
            self._set_thresholds(results=res)
            res = self._populate_thresholds(results=res)
            self.result = Result(
                results_data=res,
                simple_stats_metric=self.simple_stats_metric,
                timestamp_column_name=self.timestamp_column_name,
                chunker=self.chunker,
            )
        else:
            # TODO: review subclassing setup => superclass + '_filter' is screwing up typing.
            #       Dropping the intermediate '_filter' and directly returning the correct 'Result' class works OK
            #       but this causes us to lose the "common behavior" in the top level 'filter' method when overriding.
            #       Applicable here but to many of the base classes as well (e.g. fitting and calculating)
            self.result = self.result.filter(period='reference')
            res = self._populate_thresholds(results=res)
            self.result.data = pd.concat([self.result.data, res], ignore_index=True)

        return self.result

    def _calculate_for_df(self, data: pd.DataFrame) -> Dict[str, Any]:
        result = {}
        value = self._calculate_count_value_stats(data)
        result['value'] = value
        return result

    def _set_thresholds(self, results: pd.DataFrame):
        self._lower_alert_threshold, self._upper_alert_threshold = calculate_threshold_values(
            threshold=self.threshold,
            data=results[(self.simple_stats_metric, 'value')].to_numpy(),
            lower_threshold_value_limit=self.lower_threshold_value_limit,
            upper_threshold_value_limit=self.upper_threshold_value_limit,
            override_using_none=True,
            logger=self._logger,
            metric_name=self.simple_stats_metric,
        )

    def _populate_thresholds(self, results: pd.DataFrame):
        results[(self.simple_stats_metric, 'upper_threshold')] = self._upper_alert_threshold
        results[(self.simple_stats_metric, 'lower_threshold')] = self._lower_alert_threshold

        lower_threshold = float('-inf') if self._lower_alert_threshold is None else self._lower_alert_threshold
        upper_threshold = float('inf') if self._upper_alert_threshold is None else self._upper_alert_threshold
        results[(self.simple_stats_metric, 'alert')] = results.apply(
            lambda row: not (lower_threshold < row[(self.simple_stats_metric, 'value')] < upper_threshold),
            axis=1,
        )
        return results


def _create_multilevel_index(
    column0,
):
    chunk_column_names = ['key', 'chunk_index', 'start_index', 'end_index', 'start_date', 'end_date', 'period']
    chunk_tuples = [('chunk', chunk_column_name) for chunk_column_name in chunk_column_names]
    count_tuples = [(column0, el) for el in ['value', ]]
    tuples = chunk_tuples + count_tuples
    return MultiIndex.from_tuples(tuples)
