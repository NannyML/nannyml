#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing ways to rank drifting features."""

import abc
from typing import Dict, Optional

import pandas as pd

from nannyml.drift.base import DriftResult
from nannyml.exceptions import InvalidArgumentsException
from nannyml.metadata import ModelMetadata


class Ranking(abc.ABC):
    """Used to rank drifting features according to impact."""

    def rank(
        self,
        drift_calculation_result: DriftResult,
        model_metadata: ModelMetadata,
        only_drifting: bool = False,
    ) -> pd.DataFrame:
        """Ranks the features within a drift calculation according to impact.

        Parameters
        ----------
        drift_calculation_result : pd.DataFrame
            The drift calculation results.
        model_metadata: ModelMetadata
            Metadata describing the monitored model.
        only_drifting : bool
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

    def rank(
        self,
        drift_calculation_result: DriftResult,
        model_metadata: ModelMetadata,
        only_drifting: bool = False,
    ) -> pd.DataFrame:
        """Compares the number of alerts for each feature and uses that for ranking.

        Parameters
        ----------
        drift_calculation_result : pd.DataFrame
            The drift calculation results. Requires alert columns to be present. These are recognized and parsed
            using the ALERT_COLUMN_SUFFIX pattern, currently equal to ``'_alert'``.
        model_metadata: ModelMetadata
            Metadata describing the monitored model, used to check what the features are and exclude predictions
            from ranking results.
        only_drifting : bool
            Omits features without alerts from the ranking results.

        Returns
        -------
        feature_ranking: pd.DataFrame
            A DataFrame containing the feature names and their ranks (the highest rank starts at 1,
            second-highest rank is 2, etc.)

        Examples
        --------
        >>> import nannyml as nml
        >>> reference_df, analysis_df, target_df = nml.load_synthetic_sample()
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
            column_name
            for column_name in drift_calculation_result.data.columns
            if (self.ALERT_COLUMN_SUFFIX in column_name)
            and (model_metadata.predicted_probability_column_name not in column_name)
        ]

        if len(alert_column_names) == 0:
            raise InvalidArgumentsException('drift results are not statistical drift results.')

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

    _rankings: Dict[str, Ranking] = {'alert_count': AlertCountRanking()}

    @classmethod
    def register_ranking(cls, key: str, ranking: Ranking):
        """Registers a new calibrator to the index.

        This index associates a certain key with a Ranking instance.

        Parameters
        ----------
        key: str
            The key used to retrieve a Calibrator. When providing a key that is already in the index, the value
            will be overwritten.
        ranking: Ranking
            An instance of a Ranking subclass.

        Examples
        --------
        >>> Ranker.register_ranking('alert_count', AlertCountRanking())
        """
        cls._rankings[key] = ranking

    @classmethod
    def by(cls, key: Optional[str], **kwargs):
        """Returns a Ranking subclass instance given a key value.

        If the provided key equals ``None``, then a new instance of the default Ranking (AlertCountRanking)
        will be returned.

        If a non-existent key is provided an ``InvalidArgumentsException`` is raised.

        Parameters
        ----------
        key : str
            The key used to retrieve a Ranking. When providing a key that is already in the index, the value
            will be overwritten.

        Returns
        -------
        ranking: Ranking
            A new instance of a specific Ranking subclass.

        Examples
        --------
        >>> ranking = Ranker.by('alert_count')
        """
        default = AlertCountRanking()
        if key is None:
            return default

        if key not in cls._rankings:
            raise InvalidArgumentsException(
                f"ranking {key} unknown. " f"Please provide one of the following: {cls._rankings.keys()}"
            )

        return cls._rankings.get(key, default)
