#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing ways to rank drifting features."""

import abc
import logging
from typing import Any, Callable, Dict

import pandas as pd

from nannyml.drift.model_inputs.univariate.statistical import UnivariateStatisticalDriftCalculatorResult
from nannyml.exceptions import InvalidArgumentsException


class Ranking(abc.ABC):
    """Class that abstracts ranking features by impact on model performance."""

    def rank(
        self,
        drift_calculation_result: UnivariateStatisticalDriftCalculatorResult,
        only_drifting: bool = False,
    ) -> pd.DataFrame:
        """Ranks the features within a drift calculation according to impact on model performance.

        Parameters
        ----------
        drift_calculation_result : UnivariateStatisticalDriftCalculatorResult
            The drift calculation results.
        only_drifting : bool
            Omits non-drifting features from the ranking if True.

        Returns
        -------
        feature_ranking: pd.DataFrame
            A DataFrame containing at least a feature name and a rank per row.

        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self(**kwargs)


class Ranker:
    """Factory class to easily access Ranking implementations."""

    registry: Dict[str, Ranking] = {}

    @classmethod
    def _logger(cls) -> logging.Logger:
        return logging.getLogger(__name__)

    @classmethod
    def register(cls, key: str) -> Callable:
        """Adds a Ranking to the registry using the provided key.

        Just use the decorator above any :class:`~nannyml.drift.ranking.Ranking` subclass to have it automatically
        registered.

        Examples
        --------
        >>> @Ranker.register('alert_count')
        >>> class AlertCountRanking(Ranking):
        >>>     pass
        >>>
        >>> # Use the Ranking
        >>> ranker = nml.Ranker.by('alert_count')
        >>> ranked_features = ranker.rank(results, only_drifting=False)
        """

        def inner_wrapper(wrapped_class: Ranking) -> Ranking:
            if key in cls.registry:
                cls._logger().warning(f"re-registering Ranking for key='{key}'")
            cls.registry[key] = wrapped_class

            return wrapped_class

        return inner_wrapper

    @classmethod
    def by(cls, key: str = 'alert_count', ranking_args: Dict[str, Any] = None) -> Ranking:
        """Returns a Ranking subclass instance given a key value.

        If the provided key equals ``None``, then a new instance of the default Ranking (AlertCountRanking)
        will be returned.

        If a non-existent key is provided an ``InvalidArgumentsException`` is raised.

        Parameters
        ----------
        key : str, default='alert_count'
            The key used to retrieve a Ranking. When providing a key that is already in the index, the value
            will be overwritten.
        ranking_args: Dict[str, Any], default=None
            A dictionary of arguments that will be passed to the Ranking during creation.

        Returns
        -------
        ranking: Ranking
            A new instance of a specific Ranking subclass.

        Examples
        --------
        >>> ranking = Ranker.by('alert_count')
        """
        if ranking_args is None:
            ranking_args = {}

        if key not in cls.registry:
            raise InvalidArgumentsException(
                f"ranking {key} unknown. " f"Please provide one of the following: {cls.registry.keys()}"
            )

        ranking_class = cls.registry[key]
        return ranking_class(**ranking_args)


@Ranker.register('alert_count')
class AlertCountRanking(Ranking):
    """Ranks features by the number of drift 'alerts' they've caused."""

    ALERT_COLUMN_SUFFIX = '_alert'

    def rank(
        self,
        drift_calculation_result: UnivariateStatisticalDriftCalculatorResult,
        only_drifting: bool = False,
    ) -> pd.DataFrame:
        """Compares the number of alerts for each feature and ranks them accordingly.

        Parameters
        ----------
        drift_calculation_result : pd.DataFrame
            The drift calculation results. Requires alert columns to be present. These are recognized and parsed
            using the ALERT_COLUMN_SUFFIX pattern, currently equal to ``'_alert'``.
        only_drifting : bool, default=False
            Omits features without alerts from the ranking results.

        Returns
        -------
        feature_ranking: pd.DataFrame
            A DataFrame containing the feature names and their ranks (the highest rank starts at 1,
            second-highest rank is 2, etc.)

        Examples
        --------
        >>> import nannyml as nml
        >>> from IPython.display import display
        >>>
        >>> reference_df = nml.load_synthetic_binary_classification_dataset()[0]
        >>> analysis_df = nml.load_synthetic_binary_classification_dataset()[1]
        >>> target_df = nml.load_synthetic_binary_classification_dataset()[2]
        >>>
        >>> display(reference_df.head())
        >>>
        >>> feature_column_names = [
        >>>     col for col in reference_df.columns if col not in ['timestamp', 'y_pred_proba', 'period',
        >>>                                                        'y_pred', 'repaid']]
        >>>
        >>> calc = nml.UnivariateStatisticalDriftCalculator(feature_column_names=feature_column_names,
        >>>                                                 timestamp_column_name='timestamp')
        >>>
        >>> calc.fit(reference_df)
        >>>
        >>> results = calc.calculate(analysis_df.merge(target_df, on='identifier'))
        >>>
        >>> ranker = nml.Ranker.by('alert_count')
        >>> ranked_features = ranker.rank(results, only_drifting=False)
        >>> display(ranked_features)
                              feature  number_of_alerts  rank
        0                  identifier                10     1
        1        distance_from_office                 5     2
        2                salary_range                 5     3
        3  public_transportation_cost                 5     4
        4            wfh_prev_workday                 5     5
        5                      tenure                 2     6
        6         gas_price_per_litre                 0     7
        7                     workday                 0     8
        8            work_home_actual                 0     9
        """
        if drift_calculation_result.data.empty:
            raise InvalidArgumentsException('drift results contain no data to use for ranking')

        alert_column_names = [
            f'{name}{self.ALERT_COLUMN_SUFFIX}' for name in drift_calculation_result.calculator.feature_column_names
        ]

        ranking = pd.DataFrame(drift_calculation_result.data[alert_column_names].sum()).reset_index()
        ranking.columns = ['feature', 'number_of_alerts']
        ranking['feature'] = ranking['feature'].str.replace(self.ALERT_COLUMN_SUFFIX, '')
        ranking = ranking.sort_values('number_of_alerts', ascending=False, ignore_index=True)
        ranking['rank'] = ranking.index + 1
        if only_drifting:
            ranking = ranking.loc[ranking['number_of_alerts'] != 0, :]
        return ranking
