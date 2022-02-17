#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing ways to rank drifting features."""

import abc

import pandas as pd

from nannyml.exceptions import InvalidArgumentsException


class Ranking(abc.ABC):
    def rank(self, drift_calculation_result: pd.DataFrame, exclude_non_drifting: bool = True) -> pd.DataFrame:
        raise NotImplementedError


class AlertCountRanking(Ranking):

    _ALERT_COLUMN_SUFFIX = '_alert'

    def rank(self, drift_calculation_result: pd.DataFrame, exclude_non_drifting: bool = True) -> pd.DataFrame:

        if drift_calculation_result.empty:
            raise InvalidArgumentsException('drift results contain no data to use for ranking')

        alert_column_names = [
            column_name for column_name in drift_calculation_result.columns if self._ALERT_COLUMN_SUFFIX in column_name
        ]
        ranking = pd.DataFrame(drift_calculation_result[alert_column_names].sum()).reset_index()
        ranking.columns = ['feature', 'number_of_alerts']
        ranking['feature'] = ranking['feature'].str.replace(self._ALERT_COLUMN_SUFFIX, '')
        ranking = ranking.sort_values('number_of_alerts', ascending=False, ignore_index=True)
        ranking['rank'] = ranking.index + 1
        return ranking
