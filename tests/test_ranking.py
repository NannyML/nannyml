#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit tests for drift ranking."""

import pandas as pd
import pytest

from nannyml.drift import UnivariateDriftResult
from nannyml.drift.ranking import AlertCountRanking
from nannyml.exceptions import InvalidArgumentsException
from nannyml.metadata import ModelMetadata, extract_metadata


@pytest.fixture
def sample_drift_result() -> UnivariateDriftResult:  # noqa: D103
    return UnivariateDriftResult(
        analysis_data=[],
        model_metadata=ModelMetadata(),
        drift_data=pd.DataFrame(
            {
                'f1_alert': [0, 0, 0, 0, 1, 1],
                'f2_alert': [0, 0, 0, 1, 1, 1],
                'f3_alert': [0, 0, 0, 1, 0, 1],
                'f4_alert': [0, 0, 0, 0, 0, 0],
                'f1': [0, 0, 0, 0, 0, 0],
                'f2': [1, 1, 1, 1, 1, 1],
            }
        ),
    )


@pytest.fixture
def sample_metadata(sample_drift_result):  # noqa: D103
    md = extract_metadata(sample_drift_result.data)
    md.predicted_probability_column_name = 'y_pred_proba'
    return md


def test_alert_count_ranking_raises_invalid_arguments_exception_when_drift_result_is_empty(  # noqa: D103
    sample_metadata,
):
    ranking = AlertCountRanking()
    with pytest.raises(InvalidArgumentsException, match='drift results contain no data to use for ranking'):
        ranking.rank(
            UnivariateDriftResult(
                analysis_data=[], drift_data=pd.DataFrame(columns=['f1', 'f1_alert']), model_metadata=sample_metadata
            ),
            sample_metadata,
        )


def test_alert_count_ranking_contains_rank_column(sample_drift_result, sample_metadata):  # noqa: D103
    ranking = AlertCountRanking()
    sut = ranking.rank(sample_drift_result, sample_metadata)
    assert 'rank' in sut.columns


def test_alert_count_ranks_by_sum_of_alerts_per_feature(sample_drift_result, sample_metadata):  # noqa: D103
    ranking = AlertCountRanking()
    sut = ranking.rank(sample_drift_result, sample_metadata)
    assert sut.loc[sut['rank'] == 1, 'feature'].values[0] == 'f2'
    assert sut.loc[sut['rank'] == 2, 'feature'].values[0] == 'f1'
    assert sut.loc[sut['rank'] == 3, 'feature'].values[0] == 'f3'
    assert sut.loc[sut['rank'] == 4, 'feature'].values[0] == 'f4'


def test_alert_count_ranking_should_exclude_zero_alert_features_when_exclude_option_set(  # noqa: D103
    sample_drift_result,
    sample_metadata,
):
    ranking = AlertCountRanking()
    sut = ranking.rank(sample_drift_result, sample_metadata, only_drifting=True)
    assert len(sut) == 3
    assert len(sut[sut['feature'] == 'f4']) == 0


def test_alert_count_ranking_should_raise_invalid_arguments_exception_when_given_wrong_drift_results(  # noqa: D103
    sample_metadata,
):
    with pytest.raises(InvalidArgumentsException, match="drift results are not statistical drift results."):
        _ = AlertCountRanking().rank(
            UnivariateDriftResult(
                analysis_data=[],
                drift_data=pd.DataFrame({'alert': [False, False, True]}),
                model_metadata=sample_metadata,
            ),
            sample_metadata,
        )
