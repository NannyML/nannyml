#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing ways to rank features according to drift.

This model allows you to rank the columns within a
:class:`~nannyml.drift.univariate.calculator.UnivariateDriftCalculator` result according to their degree of drift.

The following rankers are currently available:

- :class:`~nannyml.drift.ranker.AlertCountRanker`: ranks the features according
  to the number of drift detection alerts they cause.
- :class:`~nannyml.drift.ranker.CorrelationRanker`: ranks the features according to their correlation with changes
  in realized or estimated performance.

"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from nannyml._typing import Metric
from nannyml.drift.univariate.result import Result as UnivariateResults
from nannyml.exceptions import InvalidArgumentsException, NotFittedException
from nannyml.performance_calculation.result import Result as PerformanceCalculationResults
from nannyml.performance_estimation.confidence_based.metrics import Metric as CBPEMetric
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
    """Ranks the features according to the number of drift detection alerts they cause."""

    @log_usage(UsageEvent.RANKER_ALERT_COUNT_RUN)
    def rank(
        self,
        drift_calculation_result: UnivariateResults,
        only_drifting: bool = False,
    ) -> pd.DataFrame:
        """Ranks the features according to the number of drift detection alerts they cause.

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
            second-highest rank is 2, etc.). Features with the same number of alerts are ranked alphanumerically on
            the feature name.

        Examples
        --------
        >>> import nannyml as nml
        >>> from IPython.display import display
        >>>
        >>> reference_df, analysis_df, target_df = nml.load_synthetic_binary_classification_dataset()
        >>>
        >>> display(reference_df.head())
        >>>
        >>> column_names = [
        >>>     col for col in reference_df.columns if col not in ['timestamp', 'y_pred_proba', 'period',
        >>>                                                        'y_pred', 'work_home_actual', 'identifier']]
        >>>
        >>> calc = nml.UnivariateDriftCalculator(column_names=column_names,
        >>>     timestamp_column_name='timestamp')
        >>>
        >>> calc.fit(reference_df)
        >>>
        >>> results = calc.calculate(analysis_df.merge(target_df, on='identifier'))
        >>>
        >>> ranker = nml.AlertCountRanker()
        >>> ranked_features = ranker.rank(drift_calculation_result=results, only_drifting=False)
        >>> display(ranked_features)
                number_of_alerts                 column_name  rank
        0                      5            wfh_prev_workday     1
        1                      5                salary_range     2
        2                      5  public_transportation_cost     3
        3                      5        distance_from_office     4
        4                      0                     workday     5
        5                      0            work_home_actual     6
        6                      0                      tenure     7
        7                      0         gas_price_per_litre     8
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
    """Ranks the features according to their correlation with changes in realized or estimated performance.

    Examples
        --------
        >>> import nannyml as nml
        >>> from IPython.display import display
        >>>
        >>> reference_df, analysis_df, target_df = nml.load_synthetic_binary_classification_dataset()
        >>>
        >>> column_names = [col for col in reference_df.columns
        >>>                 if col not in ['timestamp', 'y_pred_proba', 'period',
        >>>                                'y_pred', 'work_home_actual', 'identifier']]
        >>>
        >>> univ_calc = nml.UnivariateDriftCalculator(column_names=column_names,
        >>>                                           timestamp_column_name='timestamp')
        >>>
        >>> calc = nml.UnivariateDriftCalculator(column_names=column_names,
        >>>                                      timestamp_column_name='timestamp')
        >>>
        >>> univ_calc.fit(reference_df)
        >>> univariate_results = calc.calculate(analysis_df.merge(target_df, on='identifier'))
        >>>
        >>> realized_calc = nml.PerformanceCalculator(
        >>>     y_pred_proba='y_pred_proba',
        >>>     y_pred='y_pred',
        >>>     y_true='work_home_actual',
        >>>     timestamp_column_name='timestamp',
        >>>     problem_type='classification_binary',
        >>>     metrics=['roc_auc'])
        >>> realized_calc.fit(reference_df)
        >>> realized_perf_results = realized_calc.calculate(analysis_df.merge(target_df, on='identifier'))
        >>>
        >>> ranker = nml.CorrelationRanker()
        >>> # ranker fits on one metric and reference period data only
        >>> ranker.fit(realized_perf_results.filter(period='reference'))
        >>> # ranker ranks on one drift method and one performance metric
        >>> correlation_ranked_features = ranker.rank(
        >>>     univariate_results,
        >>>     realized_perf_results,
        >>>     only_drifting = False)
        >>> display(correlation_ranked_features)
                          column_name  pearsonr_correlation  pearsonr_pvalue  has_drifted  rank
        0            wfh_prev_workday              0.929710     3.076474e-09         True     1
        1  public_transportation_cost              0.925910     4.872173e-09         True     2
        2                salary_range              0.921556     8.014868e-09         True     3
        3        distance_from_office              0.920749     8.762147e-09         True     4
        4         gas_price_per_litre              0.340076     1.423541e-01        False     5
        5                     workday              0.154622     5.151128e-01        False     6
        6            work_home_actual             -0.030899     8.971071e-01        False     7
        7                      tenure             -0.177018     4.553046e-01        False     8
    """

    def __init__(self) -> None:
        """Creates a new CorrelationRanker instance."""
        super().__init__()

        self.metric: Metric
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
        """Calculates the average performance during the reference period.
        This value is saved at the `mean_reference_performance` property of the ranker.

        Parameters
        ----------
        reference_performance_calculation_result : Union[CBPEResults, DLEResults, PerformanceCalculationResults]
            Results from any performance calculator or estimator, e.g.
            :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator`
            :class:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE`
            :class:`~nannyml.performance_estimation.direct_loss_estimation.dle.DLE`

        Returns
        -------
        ranking: CorrelationRanker
        """

        if reference_performance_calculation_result is None:
            raise InvalidArgumentsException("reference performance calculation results can not be None.")
        _validate_performance_result(reference_performance_calculation_result)

        # we're expecting to have filtered inputs, so we should only have a single input.
        self.metric = reference_performance_calculation_result.metrics[0]

        # TODO: this will fail for estimated confusion matrix
        metric_column_name = self.metric.name if isinstance(self.metric, CBPEMetric) else self.metric.column_name

        self.mean_reference_performance = (
            reference_performance_calculation_result.to_df().loc[:, (metric_column_name, 'value')].mean()
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
        """Compares the number of alerts for each feature and ranks them accordingly.

        Parameters
        ----------
        drift_calculation_result: UnivariateResults
            The univariate drift results containing the features we want to rank.
        performance_calculation_result: Union[CBPEResults, DLEResults, PerformanceCalculationResults]
            Results from any performance calculator or estimator, e.g.
            :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator`
            :class:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE`
            :class:`~nannyml.performance_estimation.direct_loss_estimation.dle.DLE`
        only_drifting: bool, default=False
            Omits features without alerts from the ranking results.

        Returns
        -------
        ranking: pd.DataFrame
            A DataFrame containing the feature names and their ranks (the highest rank starts at 1,
            second-highest rank is 2, etc.). Features with the same number of alerts are ranked alphanumerically on
            the feature name.
        """
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

        # TODO: this will fail for estimated confusion matrix
        metric_column_name = self.metric.name if isinstance(self.metric, CBPEMetric) else self.metric.column_name

        # Start ranking calculations
        abs_perf_change = np.abs(
            performance_calculation_result.to_df().loc[:, (metric_column_name, 'value')].to_numpy()
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
