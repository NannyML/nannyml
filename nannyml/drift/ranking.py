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
    """Used to rank drifting features according to impact."""

    def rank(
        self,
        drift_calculation_result: UnivariateStatisticalDriftCalculatorResult,
        only_drifting: bool = False,
    ) -> pd.DataFrame:
        """Ranks the features within a drift calculation according to impact.

        Parameters
        ----------
        drift_calculation_result : pd.DataFrame
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


class AlertCountRanking(Ranking):
    """Ranks drifting features by the number of 'alerts' they've caused."""

    ALERT_COLUMN_SUFFIX = '_alert'

    def rank(
        self,
        drift_calculation_result: UnivariateStatisticalDriftCalculatorResult,
        only_drifting: bool = False,
    ) -> pd.DataFrame:
        """Compares the number of alerts for each feature and uses that for ranking.

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
        >>> reference_df, analysis_df, target_df = nml.load_synthetic_binary_classification_dataset()
        >>> metadata = nml.extract_metadata(reference_df)
        >>> metadata.target_column_name = 'work_home_actual'
        >>> calc = nml.UnivariateStatisticalDriftCalculator(metadata, chunk_size=5000)
        >>> calc.fit(reference_df)
        >>> drift = calc.calculate(analysis_df)
        >>>
        >>> ranked = Ranker.by('alert_count').rank(drift, metadata)
        >>> ranked
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


class Ranker:
    """Factory class to easily access Ranking implementations."""

    registry: Dict[str, Ranking] = {}

    @classmethod
    def _logger(cls) -> logging.Logger:
        return logging.getLogger(__name__)

    @classmethod
    def register(cls, key: str) -> Callable:
        """Adds a Ranking to the registry using the provided key."""

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
