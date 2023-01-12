#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing ways to rank drifting features."""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from nannyml.drift.univariate.result import Result as UnivariateResults
from nannyml.exceptions import InvalidArgumentsException, NotFittedException
from nannyml.performance_calculation.result import Result as PerformanceCalculationResults
from nannyml.performance_estimation.confidence_based.results import Result as CBPEResults
from nannyml.performance_estimation.direct_loss_estimation.result import Result as DLEResults
from nannyml.usage_logging import UsageEvent, log_usage


def _validate_drift_result(drift_calculation_result: UnivariateResults):
    if not isinstance(drift_calculation_result, UnivariateResults):
        raise InvalidArgumentsException("Univariate Results object required for drift_calculation_result argument.")

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


def _validate_performance_result(performance_results: Union[CBPEResults, DLEResults, PerformanceCalculationResults]):
    """Validate Inputs before performing ranking.

    Parameters
    ----------
    performance_results: Performance Estimation or Calculation results. Can be an instance of:
            nml.performance_estimation.confidence_based.results.Result,
            nml.performance_estimation.direct_loss_estimation.result.Result,
            nml.performance_calculation.result.Result
    """

    if not isinstance(performance_results, (CBPEResults, DLEResults, PerformanceCalculationResults)):
        raise InvalidArgumentsException(
            "Estimated or Realized Performance results object required for performance_results argument."
        )

    if len(performance_results.metrics) != 1:
        raise InvalidArgumentsException(
            "Just one metric should be present in performance_results used to rank CorrelationRanker."
        )


class AlertCountRanker:
    """Ranks features by the number of drift 'alerts' they've caused."""

    @log_usage(UsageEvent.RANKER_ALERT_COUNT_RUN)
    def rank(
        self,
        drift_calculation_result: UnivariateResults,
        only_drifting: bool = False,
    ) -> pd.DataFrame:
        """Compares the number of alerts for each feature and ranks them accordingly.

        Parameters
        ----------
        drift_calculation_result : nannyml.driQft.univariate.Result
            The result of a univariate drift calculation.
        only_drifting : bool, default=False
            Omits features without alerts from the ranking results.

        Returns
        -------
        ranking: pd.DataFrame
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
        >>> ranker = AlertCountRanker(drift_calculation_result=results)
        >>> ranked_features = ranker.rank(only_drifting=False)
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
        _validate_drift_result(drift_calculation_result)

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


class CorrelationRanker:
    """Ranks features according to drift correlation with performance impact.

    Ranks the features according to the correlation of their selected drift results and
    absolute performance change from mean reference performance on selected metric.
    """

    def __init__(self) -> None:
        """Creates a new CorrelationRanker instance."""
        super().__init__()

        self.metric = None
        self.mean_reference_performance: Optional[float] = None
        self.absolute_performance_change: Optional[float] = None

        self._is_fitted: bool = False

    @log_usage(UsageEvent.RANKER_CORRELATION_FIT)
    def fit(
        self,
        reference_performance_calculation_result: Optional[
            Union[CBPEResults, DLEResults, PerformanceCalculationResults]
        ] = None,
    ) -> CorrelationRanker:
        if reference_performance_calculation_result is None:
            raise InvalidArgumentsException("reference performance calculation results can not be None.")
        _validate_performance_result(reference_performance_calculation_result)

        self.metric = reference_performance_calculation_result.metrics[0]  # type: ignore
        assert self.metric is not None
        self.mean_reference_performance = (
            reference_performance_calculation_result.to_df().loc[:, (self.metric.column_name, 'value')].mean()
        )
        self._is_fitted = True
        return self

    @log_usage(UsageEvent.RANKER_CORRELATION_RUN)
    def rank(
        self,
        drift_calculation_result: UnivariateResults,
        performance_calculation_result: Optional[Union[CBPEResults, DLEResults, PerformanceCalculationResults]] = None,
        only_drifting: bool = False,
    ):
        if not self._is_fitted or self.metric is None:
            raise NotFittedException("trying to call 'rank()' on an unfitted Ranker. Please call 'fit()' first")

        # Perform input validations
        if performance_calculation_result is None:
            raise InvalidArgumentsException("reference performance calculation results can not be None.")

        _validate_drift_result(drift_calculation_result)
        _validate_performance_result(performance_calculation_result)

        _drift_index = drift_calculation_result.to_df().loc[:, ('chunk', 'chunk', 'start_index')]
        _perf_index = performance_calculation_result.to_df().loc[:, ('chunk', 'start_index')]

        if not _drift_index.equals(_perf_index):
            raise InvalidArgumentsException(
                "Drift and Performance results need to be filtered to the same data period."
            )

        # Start ranking calculations
        abs_perf_change = np.abs(
            performance_calculation_result.to_df().loc[:, (self.metric.column_name, 'value')].to_numpy()
            - self.mean_reference_performance
        )

        self.absolute_performance_change = abs_perf_change

        features1 = []
        spearmanr1 = []
        spearmanr2 = []
        has_drifted = []

        for ftr in drift_calculation_result.column_names:
            features1.append(ftr)
            tmp1 = pearsonr(
                drift_calculation_result.to_df().loc[:, (ftr, slice(None), 'value')].to_numpy().ravel(), abs_perf_change
            )
            spearmanr1.append(tmp1[0])
            spearmanr2.append(tmp1[1])
            has_drifted.append(
                (drift_calculation_result.to_df().loc[:, (ftr, slice(None), 'alert')] == True).any()[0]  # noqa: E712
            )

        ranked = pd.DataFrame(
            {
                'column_name': features1,
                'pearsonr_correlation': spearmanr1,
                'pearsonr_pvalue': spearmanr2,
                'has_drifted': has_drifted,
            }
        )

        # we want 1st row to be most impactful feature
        ranked.sort_values('pearsonr_correlation', ascending=False, inplace=True)
        ranked.reset_index(drop=True, inplace=True)
        ranked['rank'] = ranked.index + 1

        if only_drifting:
            ranked = ranked.loc[ranked.has_drifted == True].reset_index(drop=True)  # noqa: E712

        return ranked
