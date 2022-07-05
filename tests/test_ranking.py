#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit tests for drift ranking."""

import pandas as pd
import pytest

from nannyml.drift.model_inputs.univariate.statistical import (
    UnivariateStatisticalDriftCalculator,
    UnivariateStatisticalDriftCalculatorResult,
)
from nannyml.drift.ranking import AlertCountRanking
from nannyml.exceptions import InvalidArgumentsException


@pytest.fixture
def sample_drift_result() -> UnivariateStatisticalDriftCalculatorResult:  # noqa: D103
    return UnivariateStatisticalDriftCalculatorResult(
        results_data=pd.DataFrame(
            {
                'f1_alert': [0, 0, 0, 0, 1, 1],
                'f2_alert': [0, 0, 0, 1, 1, 1],
                'f3_alert': [0, 0, 0, 1, 0, 1],
                'f4_alert': [0, 0, 0, 0, 0, 0],
                'f1': [0, 0, 0, 0, 0, 0],
                'f2': [1, 1, 1, 1, 1, 1],
                'f3': [0, 0, 0, 0, 0, 0],
                'f4': [1, 1, 1, 1, 1, 1],
            }
        ),
        calculator=UnivariateStatisticalDriftCalculator(
            timestamp_column_name='timestamp', feature_column_names=['f1', 'f2', 'f3', 'f4']
        ),
    )


def test_alert_count_ranking_raises_invalid_arguments_exception_when_drift_result_is_empty():  # noqa: D103
    ranking = AlertCountRanking()
    with pytest.raises(InvalidArgumentsException, match='drift results contain no data to use for ranking'):
        ranking.rank(
            UnivariateStatisticalDriftCalculatorResult(
                results_data=pd.DataFrame(columns=['f1', 'f1_alert']),
                calculator=UnivariateStatisticalDriftCalculator(
                    timestamp_column_name='timestamp', feature_column_names=['f1', 'f2', 'f3', 'f4']
                ),
            )
        )


def test_alert_count_ranking_contains_rank_column(sample_drift_result):  # noqa: D103
    ranking = AlertCountRanking()
    sut = ranking.rank(sample_drift_result)
    assert 'rank' in sut.columns


def test_alert_count_ranks_by_sum_of_alerts_per_feature(sample_drift_result):  # noqa: D103
    ranking = AlertCountRanking()
    sut = ranking.rank(sample_drift_result)
    assert sut.loc[sut['rank'] == 1, 'feature'].values[0] == 'f2'
    assert sut.loc[sut['rank'] == 2, 'feature'].values[0] == 'f1'
    assert sut.loc[sut['rank'] == 3, 'feature'].values[0] == 'f3'
    assert sut.loc[sut['rank'] == 4, 'feature'].values[0] == 'f4'


def test_alert_count_ranking_should_exclude_zero_alert_features_when_exclude_option_set(  # noqa: D103
    sample_drift_result,
):
    ranking = AlertCountRanking()
    sut = ranking.rank(sample_drift_result, only_drifting=True)
    assert len(sut) == 3
    assert len(sut[sut['feature'] == 'f4']) == 0


def test_alert_count_ranking_should_raise_invalid_arguments_exception_when_given_wrong_drift_results():  # noqa: D103
    with pytest.raises(InvalidArgumentsException, match="drift results contain no data to use for ranking"):
        _ = AlertCountRanking().rank(
            UnivariateStatisticalDriftCalculatorResult(
                results_data=pd.DataFrame(columns=['f1', 'f1_alert']),
                calculator=UnivariateStatisticalDriftCalculator(
                    timestamp_column_name='timestamp', feature_column_names=['f1', 'f2', 'f3', 'f4']
                ),
            )
        )
