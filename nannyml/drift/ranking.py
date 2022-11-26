#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing ways to rank drifting features."""

import abc
import logging
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from nannyml.drift.univariate.result import Result as UnivariateResults
from nannyml.performance_estimation.confidence_based.results import Result as CBPEResults
from nannyml.performance_estimation.direct_loss_estimation.result import Result as DLEResults
from nannyml.performance_calculation.result import Result as PerformanceCalculationResults
from nannyml.exceptions import InvalidArgumentsException


class Ranking(abc.ABC):
    """Class that abstracts ranking features by impact on model performance."""

    def fit(
        self,
        drift_calculation_result: UnivariateResults,
        performance_results: Union[
            CBPEResults, DLEResults, PerformanceCalculationResults
        ] = None,
    ) -> pd.DataFrame:
        """Fits the calculator so it can then ranks drifted features according to their impact.

        Parameters
        ----------
        drift_calculation_result : nannyml.drift.model_inputs.univariate.statistical.Result
            The drift calculation results.
        performance_results: Performance Estimation or Calculation results. Can be an instance of:
            nml.performance_estimation.confidence_based.results.Result,
            nml.performance_estimation.direct_loss_estimation.result.Result,
            nml.performance_calculation.result.Result

        """
        raise NotImplementedError

    def rank(
        self,
        drift_calculation_result: UnivariateResults,
        performance_results: Union[
            CBPEResults, DLEResults, PerformanceCalculationResults
        ] = None,
        only_drifting: bool = False,
    ) -> pd.DataFrame:
        """Ranks the features within a drift calculation according to impact on model performance.

        Parameters
        ----------
        drift_calculation_result : nannyml.drift.model_inputs.univariate.statistical.Result
            The drift calculation results.
        performance_results: Performance Estimation or Calculation results. Can be an instance of:
                nml.performance_estimation.confidence_based.results.Result,
                nml.performance_estimation.direct_loss_estimation.result.Result,
                nml.performance_calculation.result.Result
        only_drifting : bool
            Omits non-drifting features from the ranking if True.

        Returns
        -------
        feature_ranking: pd.DataFrame
            A DataFrame containing at least a feature name and a rank per row.

        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs): # noqa D102
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
    def by(cls, key: str = 'alert_count', ranking_args: Optional[Dict[str, Any]] = None) -> Ranking:
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

    def fit( # noqa D102
        self,
        drift_calculation_result: UnivariateResults,
        performance_results: Union[
            CBPEResults, DLEResults, PerformanceCalculationResults
        ] = None,
    ) -> pd.DataFrame:

        pass

    def _validate_ranking_inputs(  # noqa: D417
        self,
        drift_calculation_result: UnivariateResults,
    ):
        """Validate Inputs before performing ranking.

        Parameters
        ----------
        drift_calculation_result : nannyml.drift.univariate.result.Result
            The drift calculation results.
        """
        if not isinstance(drift_calculation_result, UnivariateResults):
            raise InvalidArgumentsException(
                "Univariate Results object required for drift_calculation_result argument."
            )

        if drift_calculation_result.data.empty:
            raise InvalidArgumentsException('drift results contain no data to use for ranking')

        if len(drift_calculation_result.categorical_method_names) > 1:
            raise InvalidArgumentsException(
                f"Only one categorical drift method should be present in the univariate results."
                f"\nFound: {drift_calculation_result.categorical_method_names}"
            )

        if len(drift_calculation_result.continuous_method_names) > 1:
            raise InvalidArgumentsException(
                f"Only one continuous drift method should be present in the univariate results."
                f"\nFound: {drift_calculation_result.continuous_method_names}"
            )

    def rank(
        self,
        drift_calculation_result: UnivariateResults,
        only_drifting: bool = False,
    ) -> pd.DataFrame:
        """Compares the number of alerts for each feature and ranks them accordingly.

        Parameters
        ----------
        drift_calculation_result : nannyml.drift.model_inputs.univariate.statistical.Result
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
        >>> column_names = [
        >>>     col for col in reference_df.columns if col not in ['timestamp', 'y_pred_proba', 'period',
        >>>                                                        'y_pred', 'repaid', 'identifier']]
        >>>
        >>> calc = nml.UnivariateStatisticalDriftCalculator(column_names=column_names,
        >>>                                                 timestamp_column_name='timestamp')
        >>>
        >>> calc.fit(reference_df)
        >>>
        >>> results = calc.calculate(analysis_df.merge(target_df, on='identifier'))
        >>>
        >>> ranker = nml.Ranker.by('alert_count')
        >>> ranked_features = ranker.rank(results, only_drifting=False)
        >>> display(ranked_features)
                          column_name  number_of_alerts  rank
        1        distance_from_office                 5     1
        2                salary_range                 5     2
        3  public_transportation_cost                 5     3
        4            wfh_prev_workday                 5     4
        5                      tenure                 2     5
        6         gas_price_per_litre                 0     6
        7                     workday                 0     7
        8            work_home_actual                 0     8
        """
        self._validate_ranking_inputs(drift_calculation_result)

        non_chunk = list(set(drift_calculation_result.data.columns.get_level_values(0)) - {'chunk'})
        ranking = (
            drift_calculation_result.filter(period='analysis')
            .to_df()
            .loc[:, (non_chunk, slice(None), 'alert')]
            .sum()
            .reset_index()[['level_0', 0]]
        )
        ranking = ranking.groupby('level_0').sum()
        ranking.columns = ['number_of_alerts']
        ranking['column_name'] = ranking.index
        ranking = ranking.sort_values(['number_of_alerts', 'column_name'], ascending=False)
        ranking = ranking.reset_index(drop=True)
        ranking['rank'] = ranking.index + 1
        if only_drifting:
            ranking = ranking.loc[ranking['number_of_alerts'] != 0, :]
        return ranking


@Ranker.register('correlation')
class CorrelationRanking(Ranking):
    """Ranks features according to drift correlation with performance impact.

    Ranks features according to the correlation of their selected drift results and
    absolute performance change from mean reference performance on selected metric.
    """

    def fit(
        self,
        performance_results: Union[
            CBPEResults, DLEResults, PerformanceCalculationResults
        ],
    ):
        """Fit Correlation Ranker.

        Use performance results from the reference period in order to be able to compare drift
        results with absolute performance changes from the mean performance results during the reference period.

        Parameters
        ----------
        performance_results : Performance Estimation or Performance Calculation results filtered to
            the performance metric we want to compare against drift results. Additionally results need
            to be filtered to the reference period only.
            Can be an instance of:
            nml.performance_estimation.confidence_based.results.Result,
            nml.performance_estimation.direct_loss_estimation.result.Result,
            nml.performance_calculation.result.Result

        """
        # User should filter
        metrics = performance_results.metrics
        if len(metrics) > 1:
            raise InvalidArgumentsException(
                "Only one metric should be present in performance_results used to fit CorrelationRanking."
            )

        self.metric = metrics[0]
        self.mean_perf_value = performance_results.to_df().loc[:, (self.metric.column_name, 'value')].mean()

    def _validate_ranking_inputs(  # noqa: D417
        self,
        univariate_results: UnivariateResults,
        performance_results: Union[
            CBPEResults, DLEResults, PerformanceCalculationResults
        ]
    ):
        """Validate Inputs before performing ranking.

        Parameters
        ----------
        univariate_results : nannyml.drift.univariate.result.Result
            The drift calculation results.
        performance_results: Performance Estimation or Calculation results. Can be an instance of:
                nml.performance_estimation.confidence_based.results.Result,
                nml.performance_estimation.direct_loss_estimation.result.Result,
                nml.performance_calculation.result.Result
        """
        if not isinstance(univariate_results, UnivariateResults):
            raise InvalidArgumentsException(
                "Univariate Results object required for univariate_results argument."
            )

        if not isinstance(performance_results, (CBPEResults, DLEResults, PerformanceCalculationResults)):
            raise InvalidArgumentsException(
                "Estimated or Realized Performance results object required for performance_results argument."
            )

        _univ_index = univariate_results.to_df().loc[:, ('chunk', 'chunk', 'start_index')]
        _perf_index = performance_results.to_df().loc[:, ('chunk', 'start_index')]

        if not _univ_index.equals(_perf_index):
            raise InvalidArgumentsException(
                "Drift and Performance results need to be filtered to the same data period.")

        if len(univariate_results.categorical_method_names) > 1:
            raise InvalidArgumentsException(
                f"Only one categorical drift method should be present in the univariate results."
                f"\nFound: {univariate_results.categorical_method_names}"
            )

        if len(univariate_results.continuous_method_names) > 1:
            raise InvalidArgumentsException(
                f"Only one continuous drift method should be present in the univariate results."
                f"\nFound: {univariate_results.continuous_method_names}"
            )

        if not hasattr(self, 'metric'):
            raise InvalidArgumentsException("CorrelationRanking needs to call fit method before rank.")

        if len(performance_results.metrics) > 1:
            raise InvalidArgumentsException(
                "Only one metric should be present in performance_results used to rank CorrelationRanking.")

        metric = performance_results.metrics[0]
        if not isinstance(metric, type(self.metric)):
            raise InvalidArgumentsException(
                "Performance results need to be filtered with the same metric"
                " for fit and rank methods of Correlation Ranker."
            )

    def rank(  # noqa: D102
        self,
        univariate_results: UnivariateResults,
        performance_results: Union[
            CBPEResults, DLEResults, PerformanceCalculationResults
        ],
        only_drifting: bool = False,

    ):

        self._validate_ranking_inputs(univariate_results, performance_results)

        abs_perf_change = np.abs(
            performance_results.to_df().loc[:, (self.metric.column_name, 'value')].to_numpy() - self.mean_perf_value
        )
        self.abs_perf_change = abs_perf_change

        features1 = []
        spearmanr1 = []
        spearmanr2 = []
        has_drifted = []

        for ftr in univariate_results.column_names:
            features1.append(ftr)
            tmp1 = pearsonr(
                univariate_results.to_df().loc[:, (ftr, slice(None), 'value')].to_numpy().ravel(),
                abs_perf_change
            )
            spearmanr1.append(
                tmp1[0]
            )
            spearmanr2.append(
                tmp1[1]
            )
            has_drifted.append(
                (univariate_results.to_df().loc[:, (ftr, slice(None), 'alert')] == True).any()[0]  # noqa: E712
            )

        ranked = pd.DataFrame({
            'column_name': features1,
            'pearsonr_correlation': spearmanr1,
            'pearsonr_pvalue': spearmanr2,
            'has_drifted': has_drifted
        })

        # we want 1st row to be most impactful feature
        ranked.sort_values('pearsonr_correlation', ascending=False, inplace=True)
        ranked.reset_index(drop=True, inplace=True)
        ranked['rank'] = ranked.index + 1
        ranked

        if only_drifting:
            ranked = ranked.loc[ranked.has_drifted == True].reset_index(drop=True)  # noqa: E712

        return ranked
