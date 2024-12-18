#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

"""Simple Statistics Average Calculator."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pandas import MultiIndex

from nannyml.base import AbstractCalculator, _list_missing, _split_features_by_type
from nannyml.chunk import Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.sampling_error import SAMPLING_ERROR_RANGE
from nannyml.stats.avg.result import Result
from nannyml.thresholds import StandardDeviationThreshold, Threshold, calculate_threshold_values
from nannyml.usage_logging import UsageEvent, log_usage


class SummaryStatsAvgCalculator(AbstractCalculator):
    """SummaryStatsAvgCalculator implementation."""

    def __init__(
        self,
        column_names: Union[str, List[str]],
        timestamp_column_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_number: Optional[int] = None,
        chunk_period: Optional[str] = None,
        chunker: Optional[Chunker] = None,
        threshold: Threshold = StandardDeviationThreshold(),
    ):
        """Creates a new SummaryStatsAvgCalculator instance.

        Parameters
        ----------
        column_names: Union[str, List[str]]
            A string or list containing the names of features in the provided data set.
            Missing Values will be calculated for each entry in this list.
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
        >>> column_names = ['car_value', 'debt_to_income_ratio', 'driver_tenure']
        >>> calc = nml.SummaryStatsSumCalculator(
        ...     column_names=column_names,
        ...     timestamp_column_name='timestamp',
        ... ).fit(reference)
        >>> res = calc.calculate(analysis)
        >>> for column_name in res.column_names:
        ...     res = res.filter(period='analysis', column_name=column_name).plot().show()
        """
        super(SummaryStatsAvgCalculator, self).__init__(
            chunk_size, chunk_number, chunk_period, chunker, timestamp_column_name
        )
        if isinstance(column_names, str):
            self.column_names = [column_names]
        elif isinstance(column_names, list):
            for el in column_names:
                if not isinstance(el, str):
                    raise InvalidArgumentsException(
                        f"column_names elements should be either a column name string or a list of strings, found\n{el}"
                    )
            self.column_names = column_names
        else:
            raise InvalidArgumentsException(
                "column_names should be either a column name string or a list of columns names strings, "
                f"found {column_names}"
            )

        self.result: Optional[Result] = None
        # Vanilla standard error of the mean applies here.
        self._sampling_error_components: Dict[str, float] = {column_name: 0 for column_name in self.column_names}
        # threshold strategy is the same across all columns
        self.threshold = threshold
        self._upper_alert_thresholds: Dict[str, Optional[float]] = {column_name: 0 for column_name in self.column_names}
        self._lower_alert_thresholds: Dict[str, Optional[float]] = {column_name: 0 for column_name in self.column_names}

        self.lower_threshold_value_limit: Optional[float] = None
        self.upper_threshold_value_limit: Optional[float] = None
        self.simple_stats_metric = 'values_avg'

    @log_usage(UsageEvent.STATS_AVG_FIT)
    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs):
        """Fits the drift calculator to a set of reference data."""
        if reference_data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing(self.column_names, reference_data)

        continuous_column_names, categorical_column_names = _split_features_by_type(reference_data, self.column_names)
        if len(categorical_column_names) >= 1:
            raise InvalidArgumentsException(
                f"Cannot calculate average for categorical columns:\n {categorical_column_names}"
            )

        for col in self.column_names:
            self._sampling_error_components[col] = reference_data[col].std()

        self.result = self._calculate(data=reference_data)
        self.result.data[('chunk', 'period')] = 'reference'

        return self

    @log_usage(UsageEvent.STATS_AVG_RUN)
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> Result:
        """Calculates methods for both categorical and continuous columns."""
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing(self.column_names, data)

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

            for column_name in self.column_names:
                for k, v in self._calculate_for_column(chunk.data, column_name).items():
                    row[f'{column_name}_{k}'] = v

            rows.append(row)

        result_index = _create_multilevel_index(
            column_names=self.column_names,
        )
        res = pd.DataFrame(rows)
        res.columns = result_index
        res = res.reset_index(drop=True)

        if self.result is None:
            self._set_thresholds(results=res)
            res = self._populate_thresholds(results=res)
            self.result = Result(
                results_data=res,
                column_names=self.column_names,
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

    def _calculate_for_column(self, data: pd.DataFrame, column_name: str) -> Dict[str, Any]:
        result = {}
        try:
            value = _calculate_avg_value_stats(data[column_name])
            result['value'] = value
            result['sampling_error'] = self._sampling_error_components[column_name] / np.sqrt(data.shape[0])
            result['upper_confidence_boundary'] = result['value'] + SAMPLING_ERROR_RANGE * result['sampling_error']
            result['lower_confidence_boundary'] = result['value'] - SAMPLING_ERROR_RANGE * result['sampling_error']
        except Exception as exc:
            if self._logger:
                self._logger.error(
                    f"an unexpected exception occurred during calculation of column '{column_name}': " f"{exc}"
                )
            result['value'] = np.nan
            result['sampling_error'] = np.nan
            result['upper_confidence_boundary'] = np.nan
            result['lower_confidence_boundary'] = np.nan
        finally:
            return result

    def _set_thresholds(self, results: pd.DataFrame):
        for column in self.column_names:
            self._lower_alert_thresholds[column], self._upper_alert_thresholds[column] = calculate_threshold_values(
                threshold=self.threshold,
                data=results[(column, 'value')].to_numpy(),
                lower_threshold_value_limit=self.lower_threshold_value_limit,
                upper_threshold_value_limit=self.upper_threshold_value_limit,
                override_using_none=True,
                logger=self._logger,
                metric_name=column,
            )

    def _populate_thresholds(self, results: pd.DataFrame):
        for column in self.column_names:
            results[(column, 'upper_threshold')] = self._upper_alert_thresholds[column]
            results[(column, 'lower_threshold')] = self._lower_alert_thresholds[column]

            lower_threshold = float('-inf') if self._lower_alert_thresholds[column] is None else self._lower_alert_thresholds[column]  # noqa: E501
            upper_threshold = float('inf') if self._upper_alert_thresholds[column] is None else self._upper_alert_thresholds[column]  # noqa: E501
            results[(column, 'alert')] = results.apply(
                lambda row: not (lower_threshold < row[(column, 'value')] < upper_threshold),
                axis=1,
            )
        return results


def _create_multilevel_index(
    column_names,
):
    chunk_column_names = ['key', 'chunk_index', 'start_index', 'end_index', 'start_date', 'end_date', 'period']
    chunk_tuples = [('chunk', chunk_column_name) for chunk_column_name in chunk_column_names]
    column_tuples = [
        (column_name, el)
        for column_name in column_names
        for el in [
            'value',
            'sampling_error',
            'upper_confidence_boundary',
            'lower_confidence_boundary',
        ]
    ]
    tuples = chunk_tuples + column_tuples
    return MultiIndex.from_tuples(tuples)


def _calculate_avg_value_stats(data: pd.Series):
    return data.mean()
