#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit tests for drift ranking."""
import copy

import pandas as pd
import pytest

from nannyml.datasets import load_synthetic_binary_classification_dataset
from nannyml.drift.ranking import Ranker, AlertCountRanking, CorrelationRanking
from nannyml.drift.univariate import Result, UnivariateDriftCalculator
from nannyml.exceptions import InvalidArgumentsException


@pytest.fixture(scope='module')
def sample_drift_result() -> Result:  # noqa: D103
    reference, analysis, _ = load_synthetic_binary_classification_dataset()
    calc = UnivariateDriftCalculator(
        timestamp_column_name='timestamp',
        column_names=[
            col for col in reference.columns if col not in ['timestamp', 'identifier', 'work_home_actual', 'period']
        ],
        continuous_methods=['kolmogorov_smirnov', 'jensen_shannon'],
        chunk_size=5000,
    ).fit(reference)
    result = calc.calculate(analysis)
    return result

def test_alertcount_ranker_creation():
    ranker = Ranker.by('alert_count')
    assert isinstance(ranker, AlertCountRanking)

def test_correlation_ranker_creation():
    ranker = Ranker.by('correlation')
    assert isinstance(ranker, CorrelationRanking)


def test_alert_count_ranking_raises_invalid_arguments_exception_when_drift_result_is_empty(
    sample_drift_result,
):  # noqa: D103
    ranking = AlertCountRanking()
    result = copy.deepcopy(sample_drift_result)
    result.data = pd.DataFrame(columns=['f1', 'f2', 'f3', 'f4'])
    with pytest.raises(InvalidArgumentsException, match='drift results contain no data'):
        ranking.rank(result)


def test_alert_count_ranking_contains_rank_column(sample_drift_result):  # noqa: D103
    ranking = AlertCountRanking()
    sut = ranking.rank(sample_drift_result)
    assert 'rank' in sut.columns


def test_alert_count_ranks_by_sum_of_alerts_per_feature(sample_drift_result):  # noqa: D103
    ranking = AlertCountRanking()
    sut = ranking.rank(sample_drift_result)
    assert sut.loc[sut['rank'] == 1, 'column_name'].values[0] == 'y_pred_proba'
    assert sut.loc[sut['rank'] == 2, 'column_name'].values[0] == 'public_transportation_cost'
    assert sut.loc[sut['rank'] == 3, 'column_name'].values[0] == 'distance_from_office'
    assert sut.loc[sut['rank'] == 4, 'column_name'].values[0] == 'y_pred'


def test_alert_count_ranking_should_exclude_zero_alert_features_when_exclude_option_set(  # noqa: D103
    sample_drift_result,
):
    ranking = AlertCountRanking()
    sut = ranking.rank(sample_drift_result, only_drifting=True)
    assert len(sut) == 7
    assert not any(sut['column_name'] == 'gas_price_per_litre')
