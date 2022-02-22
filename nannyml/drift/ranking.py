#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing ways to rank drifting features."""

import abc

import pandas as pd

from nannyml.exceptions import InvalidArgumentsException


def rank_drifted_features(drift_calculation_result: pd.DataFrame, by: str = 'alert_count') -> pd.DataFrame:
    """Used to rank drifted features according to multiple strategies.

    Parameters
    ----------
    drift_calculation_result : pd.DataFrame
        The drift calculation result containing the features to rank.
    by : str, default='alert_count'
        The strategy to use for ranking.
        Must be one of these values: `alert_count`

    Returns
    -------
    ranking: pd.DataFrame
        A DataFrame containing a feature name and a rank per row.

    """
    if by == 'alert_count':
        return AlertCountRanking().rank(drift_calculation_result=drift_calculation_result)
    else:
        raise InvalidArgumentsException(f"unknown value '{by}'. Provide one of the following: ['alert_count']")


class Ranking(abc.ABC):
    """Used to rank drifting features according to impact."""

    def rank(self, drift_calculation_result: pd.DataFrame, exclude_non_drifting: bool = False) -> pd.DataFrame:
        """Ranks the features within a drift calculation according to impact.

        Parameters
        ----------
        drift_calculation_result : pd.DataFrame
            The drift calculation results.
        exclude_non_drifting : bool
            Omits non-drifting features from the ranking if True.

        Returns
        -------
        feature_ranking: pd.DataFrame
            A DataFrame containing at least a feature name and a rank per row.

        """
        raise NotImplementedError


class AlertCountRanking(Ranking):
    """Ranks drifting features by the number of 'alerts' they've caused."""

    ALERT_COLUMN_SUFFIX = '_alert'

    def rank(self, drift_calculation_result: pd.DataFrame, exclude_non_drifting: bool = False) -> pd.DataFrame:
        """Compares the number of alerts for each feature and uses that for ranking.

        Parameters
        ----------
        drift_calculation_result : pd.DataFrame
            The drift calculation results. Requires alert columns to be present. These are recognized and parsed
            using the ALERT_COLUMN_SUFFIX pattern, currently equal to ``'_alert'``.
        exclude_non_drifting : bool
            Omits features without alerts from the ranking results.

        Returns
        -------
        feature_ranking: pd.DataFrame
            A DataFrame containing the feature names and their ranks (the highest rank starts at 1,
            second-highest rank is 2, etc.)

        """
        if drift_calculation_result.empty:
            raise InvalidArgumentsException('drift results contain no data to use for ranking')

        alert_column_names = [
            column_name for column_name in drift_calculation_result.columns if self.ALERT_COLUMN_SUFFIX in column_name
        ]

        if len(alert_column_names) == 0:
            raise InvalidArgumentsException('drift results are not univariate drift results.')

        ranking = pd.DataFrame(drift_calculation_result[alert_column_names].sum()).reset_index()
        ranking.columns = ['feature', 'number_of_alerts']
        ranking['feature'] = ranking['feature'].str.replace(self.ALERT_COLUMN_SUFFIX, '')
        ranking = ranking.sort_values('number_of_alerts', ascending=False, ignore_index=True)
        ranking['rank'] = ranking.index + 1
        if exclude_non_drifting:
            ranking = ranking.loc[ranking['number_of_alerts'] != 0, :]
        return ranking
