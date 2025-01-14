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

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from nannyml._typing import Metric
from nannyml.data_quality.missing.result import Result as MissingValueResults
from nannyml.data_quality.unseen.result import Result as UnseenValuesResults
from nannyml.drift.univariate.result import Result as UnivariateResults
from nannyml.exceptions import InvalidArgumentsException, NotFittedException
from nannyml.performance_calculation.result import Result as PerformanceCalculationResults
from nannyml.performance_estimation.confidence_based.metrics import Metric as CBPEMetric
from nannyml.performance_estimation.confidence_based.results import Result as CBPEResults
from nannyml.performance_estimation.direct_loss_estimation.result import Result as DLEResults
from nannyml.stats.avg.result import Result as StatsAvgResults
from nannyml.stats.count import Result as StatsCountResults
from nannyml.stats.median import Result as StatsMedianResults
from nannyml.stats.std import Result as StatsStdResults
from nannyml.stats.sum import Result as StatsSumResults
from nannyml.usage_logging import UsageEvent, log_usage

RankableResult = Union[
    UnivariateResults,
    MissingValueResults,
    UnseenValuesResults,
    StatsAvgResults,
    StatsCountResults,
    StatsStdResults,
    StatsSumResults,
    StatsMedianResults,
]
PerformanceResults = Union[CBPEResults, DLEResults, PerformanceCalculationResults]

_logger = logging.getLogger(__name__)


def _validate_drift_result(rankable_result: RankableResult):
    if not isinstance(
        rankable_result,
        (
            UnivariateResults,
            MissingValueResults,
            UnseenValuesResults,
            StatsAvgResults,
            StatsCountResults,
            StatsStdResults,
            StatsSumResults,
            StatsMedianResults,
        ),
    ):
        raise InvalidArgumentsException(
            f"`rankable_result` should be one of `[UnivariateResults, MissingValueResults, "
            f"UnseenValuesResults, StatsAvgResults, StatsCountResults, StatsStdResults, "
            f"StatsSumResults, StatsMedianResults]`."
            f"\ngot {str(type(rankable_result))}"
        )

    if rankable_result.empty:
        raise InvalidArgumentsException('rankable_result contains no data to use for ranking')

    if isinstance(rankable_result, UnivariateResults):
        if len(rankable_result.categorical_method_names) > 1:
            raise InvalidArgumentsException(
                f"Only one categorical drift method should be present in the univariate results."
                f"\nFound: {rankable_result.categorical_method_names}"
            )

        if len(rankable_result.continuous_method_names) > 1:
            raise InvalidArgumentsException(
                f"Only one continuous drift method should be present in the univariate results."
                f"\nFound: {rankable_result.continuous_method_names}"
            )


def _validate_performance_result(performance_result: PerformanceResults):
    """Validate Inputs before performing ranking.

    Parameters
    ----------
    performance_result: Performance Estimation or Calculation results. Can be an instance of:
            nml.performance_estimation.confidence_based.results.Result,
            nml.performance_estimation.direct_loss_estimation.result.Result,
            nml.performance_calculation.result.Result
    """

    if not isinstance(performance_result, (CBPEResults, DLEResults, PerformanceCalculationResults)):
        raise InvalidArgumentsException(
            "Estimated or Realized Performance results object required for performance_results argument."
        )

    if len(performance_result.metrics) != 1:
        raise InvalidArgumentsException(
            "Just one metric should be present in performance_results used to rank CorrelationRanker."
        )


class AlertCountRanker:
    """Ranks the features according to the number of drift detection alerts they cause."""

    @log_usage(UsageEvent.RANKER_ALERT_COUNT_RUN)
    def rank(
        self,
        rankable_result: RankableResult,
        only_drifting: bool = False,
    ) -> pd.DataFrame:
        """Ranks the features according to the number of drift detection alerts they cause.

        Parameters
        ----------
        rankable_result : RankableResult
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
        >>> reference_df, analysis_df, analysis_targets_df = nml.load_synthetic_car_loan_dataset()
        >>> analysis_full_df = analysis_df.merge(analysis_targets_df, left_index=True, right_index=True)
        >>> feature_column_names = [
        ...     'car_value', 'salary_range', 'debt_to_income_ratio', 'loan_length', 'repaid_loan_on_prev_car',
        ...     'size_of_downpayment', 'driver_tenure', 'y_pred_proba', 'y_pred', 'repaid'
        >>> ]
        >>> univ_calc = nml.UnivariateDriftCalculator(
        ...     column_names=feature_column_names,
        ...     treat_as_categorical=['y_pred', 'repaid'],
        ...     timestamp_column_name='timestamp',
        ...     continuous_methods=['kolmogorov_smirnov', 'jensen_shannon'],
        ...     categorical_methods=['chi2', 'jensen_shannon'],
        ...     chunk_size=5000
        >>> )
        >>> univ_calc.fit(reference_df)
        >>> univariate_results = univ_calc.calculate(analysis_full_df)
        >>> alert_count_ranker = nml.AlertCountRanker()
        >>> alert_count_ranked_features = alert_count_ranker.rank(
        ...     univariate_results.filter(methods=['jensen_shannon']),
        ...     only_drifting = False)
        >>> display(alert_count_ranked_features)
                number_of_alerts                 column_name  rank
        0                      5                y_pred_proba     1
        1                      5                salary_range     2
        2                      5     repaid_loan_on_prev_car     3
        3                      5                 loan_length     4
        4                      0                   car_value     5
        5                      0                      y_pred     6
        6                      0         size_of_downpayment     7
        7                      0                      repaid     8
        8                      0               driver_tenure     9
        9                      0        debt_to_income_ratio     10
        """
        _validate_drift_result(rankable_result)

        key_list = rankable_result.keys()
        ranking = (
            pd.concat([rankable_result.alerts(_key) for _key in key_list], axis=1).sum().reset_index()[['level_0', 0]]
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
        >>> reference_df, analysis_df, analysis_targets_df = nml.load_synthetic_car_loan_dataset()
        >>> analysis_full_df = analysis_df.merge(analysis_targets_df, left_index=True, right_index=True)
        >>> feature_column_names = [
        ...     'car_value', 'salary_range', 'debt_to_income_ratio', 'loan_length', 'repaid_loan_on_prev_car',
        ...     'size_of_downpayment', 'driver_tenure', 'y_pred_proba', 'y_pred', 'repaid'
        >>> ]
        >>> univ_calc = nml.UnivariateDriftCalculator(
        ...     column_names=feature_column_names,
        ...     treat_as_categorical=['y_pred', 'repaid'],
        ...     timestamp_column_name='timestamp',
        ...     continuous_methods=['kolmogorov_smirnov', 'jensen_shannon'],
        ...     categorical_methods=['chi2', 'jensen_shannon'],
        ...     chunk_size=5000
        >>> )
        >>> univ_calc.fit(reference_df)
        >>> univariate_results = univ_calc.calculate(analysis_full_df)
        >>> realized_calc = nml.PerformanceCalculator(
        ...     y_pred_proba='y_pred_proba',
        ...     y_pred='y_pred',
        ...     y_true='repaid',
        ...     timestamp_column_name='timestamp',
        ...     problem_type='classification_binary',
        ...     metrics=['roc_auc', 'recall',],
        ...     chunk_size=5000)
        >>> realized_calc.fit(reference_df)
        >>> realized_perf_results = realized_calc.calculate(analysis_full_df)
        >>> ranker2 = nml.CorrelationRanker()
        >>> # ranker fits on one metric and reference period data only
        >>> ranker2.fit(
        ...     realized_perf_results.filter(period='reference', metrics=['recall']))
        >>> # ranker ranks on one drift method and one performance metric
        >>> correlation_ranked_features2 = ranker2.rank(
        ...     univariate_results.filter(period='analysis', methods=['jensen_shannon']),
        ...     realized_perf_results.filter(period='analysis', metrics=['recall']),
        ...     only_drifting = False)
        >>> display(correlation_ranked_features2)
                          column_name  pearsonr_correlation  pearsonr_pvalue  has_drifted  rank
        0     repaid_loan_on_prev_car               0.96897      3.90719e-06         True     1
        1                y_pred_proba              0.966157      5.50918e-06         True     2
        2                 loan_length              0.965298      6.08385e-06         True     3
        3                   car_value              0.963623      7.33185e-06         True     4
        4                salary_range              0.963456      7.46561e-06         True     5
        5         size_of_downpayment              0.308948         0.385072        False     6
        6        debt_to_income_ratio              0.307373         0.387627        False     7
        7                      y_pred             -0.357571         0.310383        False     8
        8                      repaid             -0.395842         0.257495        False     9
        9               driver_tenure             -0.575807        0.0815202        False     10
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
        reference_performance_calculation_result: Optional[PerformanceResults] = None,
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
        rankable_result: RankableResult,
        performance_result: Optional[PerformanceResults] = None,
        only_drifting: bool = False,
    ):
        """Compares the number of alerts for each feature and ranks them accordingly.

        Parameters
        ----------
        rankable_result: RankableResult
            The univariate, data quality or simple statistic drift results containing the features we want to rank.
        performance_result: PerformanceResults
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
        if performance_result is None:
            raise InvalidArgumentsException("reference performance calculation results can not be None.")

        _validate_drift_result(rankable_result)
        _validate_performance_result(performance_result)

        _drift_index = rankable_result.chunk_start_indices
        _perf_index = performance_result.chunk_start_indices

        if not _drift_index.equals(_perf_index):
            raise InvalidArgumentsException(
                "Drift and Performance results need to be filtered to the same data period."
            )

        # TODO: this will fail for estimated confusion matrix
        metric_column_name = self.metric.name if isinstance(self.metric, CBPEMetric) else self.metric.column_name

        # Start ranking calculations
        abs_perf_change = np.abs(
            performance_result.to_df().loc[:, (metric_column_name, 'value')].to_numpy()
            - self.mean_reference_performance
        )

        self.absolute_performance_change = abs_perf_change

        features1 = []
        spearmanr1 = []
        spearmanr2 = []
        has_drifted = []

        for _key in rankable_result.keys():
            features1.append(_key.display_names[0])
            values = rankable_result.values(_key)

            if values is None or values.empty:
                _logger.info(f"skipped ranking `None` rankable values for key '{_key}'")
                break

            # Remove NaN values
            feature_nan, perf_nan = np.isnan(values.to_numpy()), np.isnan(abs_perf_change)
            filtered_values = values[~(feature_nan | perf_nan)]
            filtered_perf_change = abs_perf_change[~(feature_nan | perf_nan)]

            tmp1 = (
                pearsonr(filtered_values.ravel(), filtered_perf_change)
                if len(filtered_values) > 1
                else (np.nan, np.nan)
            )
            spearmanr1.append(tmp1[0])
            spearmanr2.append(tmp1[1])

            alerts = rankable_result.alerts(_key)
            has_drifted.append(alerts.any() if alerts is not None else False)

        ranked = pd.DataFrame(
            {
                'column_name': features1,
                'pearsonr_correlation': spearmanr1,
                'pearsonr_pvalue': spearmanr2,
                'has_drifted': has_drifted,
            }
        )

        # we want first row to be the most impactful feature
        ranked.sort_values('pearsonr_correlation', ascending=False, inplace=True)
        ranked.reset_index(drop=True, inplace=True)
        ranked['rank'] = ranked.index + 1

        if only_drifting:
            ranked = ranked.loc[ranked.has_drifted == True].reset_index(drop=True)  # noqa: E712

        return ranked
