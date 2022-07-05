#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Tests for Drift package."""

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from nannyml.chunk import PeriodBasedChunker
from nannyml.drift.model_inputs.multivariate.data_reconstruction.calculator import (
    DataReconstructionDriftCalculator,
    _minimum_chunk_size,
)


@pytest.fixture
def sample_drift_data() -> pd.DataFrame:  # noqa: D103
    data = pd.DataFrame(pd.date_range(start='1/6/2020', freq='10min', periods=20 * 1008), columns=['timestamp'])
    data['week'] = data.timestamp.dt.isocalendar().week - 1
    data['period'] = 'reference'
    data.loc[data.week >= 11, ['period']] = 'analysis'
    # data[NML_METADATA_PERIOD_COLUMN_NAME] = data['period']  # simulate preprocessing
    np.random.seed(167)
    data['f1'] = np.random.randn(data.shape[0])
    data['f2'] = np.random.rand(data.shape[0])
    data['f3'] = np.random.randint(4, size=data.shape[0])
    data['f4'] = np.random.randint(20, size=data.shape[0])
    data['y_pred_proba'] = np.random.rand(data.shape[0])
    data['output'] = np.random.randint(2, size=data.shape[0])
    data['actual'] = np.random.randint(2, size=data.shape[0])

    # Rule 1b is the shifted feature, 75% 0 instead of 50%
    rule1a = {2: 0, 3: 1}
    rule1b = {2: 0, 3: 0}
    data.loc[data.week < 16, ['f3']] = data.loc[data.week < 16, ['f3']].replace(rule1a)
    data.loc[data.week >= 16, ['f3']] = data.loc[data.week >= 16, ['f3']].replace(rule1b)

    # Rule 2b is the shifted feature
    c1 = 'white'
    c2 = 'red'
    c3 = 'green'
    c4 = 'blue'

    rule2a = {
        0: c1,
        1: c1,
        2: c1,
        3: c1,
        4: c1,
        5: c2,
        6: c2,
        7: c2,
        8: c2,
        9: c2,
        10: c3,
        11: c3,
        12: c3,
        13: c3,
        14: c3,
        15: c4,
        16: c4,
        17: c4,
        18: c4,
        19: c4,
    }

    rule2b = {
        0: c1,
        1: c1,
        2: c1,
        3: c1,
        4: c1,
        5: c2,
        6: c2,
        7: c2,
        8: c2,
        9: c2,
        10: c3,
        11: c3,
        12: c3,
        13: c1,
        14: c1,
        15: c4,
        16: c4,
        17: c4,
        18: c1,
        19: c2,
    }

    data.loc[data.week < 16, ['f4']] = data.loc[data.week < 16, ['f4']].replace(rule2a)
    data.loc[data.week >= 16, ['f4']] = data.loc[data.week >= 16, ['f4']].replace(rule2b)

    data.loc[data.week >= 16, ['f1']] = data.loc[data.week >= 16, ['f1']] + 0.6
    data.loc[data.week >= 16, ['f2']] = np.sqrt(data.loc[data.week >= 16, ['f2']])
    data.drop(columns=['week'], inplace=True)

    data['f3'] = data['f3'].astype("category")

    return data


@pytest.fixture
def sample_drift_data_with_nans(sample_drift_data) -> pd.DataFrame:  # noqa: D103
    data = sample_drift_data.copy(deep=True)
    data['id'] = data.index
    nan_pick1 = set(data.id.sample(frac=0.11, random_state=13))
    nan_pick2 = set(data.id.sample(frac=0.11, random_state=14))
    data.loc[data.id.isin(nan_pick1), 'f1'] = np.NaN
    data.loc[data.id.isin(nan_pick2), 'f4'] = np.NaN
    data.drop(columns=['id'], inplace=True)
    return data


def test_data_reconstruction_drift_calculator_with_params_should_not_fail(sample_drift_data):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    calc = DataReconstructionDriftCalculator(
        feature_column_names=['f1', 'f2', 'f3', 'f4'],
        timestamp_column_name='timestamp',
        n_components=0.75,
        chunk_period='W',
    ).fit(ref_data)
    try:
        drift = calc.calculate(data=sample_drift_data)
        print(drift)
    except Exception:
        pytest.fail()


def test_data_reconstruction_drift_calculator_with_default_params_should_not_fail(sample_drift_data):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    calc = DataReconstructionDriftCalculator(
        feature_column_names=['f1', 'f2', 'f3', 'f4'], timestamp_column_name='timestamp'
    ).fit(ref_data)
    try:
        drift = calc.calculate(data=sample_drift_data)
        print(drift)
    except Exception:
        pytest.fail()


def test_data_reconstruction_drift_calculator_with_default_params_should_not_fail_w_nans(  # noqa: D103
    sample_drift_data_with_nans,
):
    ref_data = sample_drift_data_with_nans.loc[sample_drift_data_with_nans['period'] == 'reference']
    calc = DataReconstructionDriftCalculator(
        feature_column_names=['f1', 'f2', 'f3', 'f4'],
        timestamp_column_name='timestamp',
        n_components=0.75,
        chunk_period='W',
    ).fit(ref_data)
    try:
        drift = calc.calculate(data=sample_drift_data_with_nans)
        print(drift)
    except Exception:
        pytest.fail()


def test_data_reconstruction_drift_calculator_should_contain_chunk_details_and_single_drift_value_column(  # noqa: D103
    sample_drift_data,
):
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    calc = DataReconstructionDriftCalculator(
        feature_column_names=['f1', 'f2', 'f3', 'f4'],
        timestamp_column_name='timestamp',
        n_components=0.75,
        chunk_period='W',
    ).fit(ref_data)
    drift = calc.calculate(data=sample_drift_data)

    sut = drift.data.columns
    assert len(sut) == 9
    assert 'key' in sut
    assert 'start_index' in sut
    assert 'start_date' in sut
    assert 'end_index' in sut
    assert 'end_date' in sut
    assert 'upper_threshold' in sut
    assert 'lower_threshold' in sut
    assert 'alert' in sut
    assert 'reconstruction_error' in sut


def test_data_reconstruction_drift_calculator_should_contain_a_row_for_each_chunk(sample_drift_data):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    calc = DataReconstructionDriftCalculator(
        feature_column_names=['f1', 'f2', 'f3', 'f4'],
        timestamp_column_name='timestamp',
        n_components=0.75,
        chunk_period='W',
    ).fit(ref_data)
    drift = calc.calculate(data=sample_drift_data)

    expected = len(
        PeriodBasedChunker(offset='W').split(sample_drift_data, minimum_chunk_size=1, timestamp_column_name='timestamp')
    )
    sut = len(drift.data)
    assert sut == expected


# TODO: find a better way to test this
def test_data_reconstruction_drift_calculator_should_not_fail_when_using_feature_subset(  # noqa: D103
    sample_drift_data,
):
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']

    calc = DataReconstructionDriftCalculator(feature_column_names=['f1', 'f3'], timestamp_column_name='timestamp').fit(
        ref_data
    )
    try:
        calc.fit(ref_data)
        calc.calculate(sample_drift_data)
    except Exception as exc:
        pytest.fail(f"should not have failed but got {exc}")


def test_data_reconstruction_drift_calculator_numeric_results(sample_drift_data):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']

    # calc = DataReconstructionDriftCalculator(sample_drift_metadata, chunk_period='W').fit(ref_data)
    calc = DataReconstructionDriftCalculator(
        feature_column_names=['f1', 'f2', 'f3', 'f4'], timestamp_column_name='timestamp', chunk_period='W'
    ).fit(ref_data)
    drift = calc.calculate(data=sample_drift_data)
    expected_drift = pd.DataFrame.from_dict(
        {
            'key': [
                '2020-01-06/2020-01-12',
                '2020-01-13/2020-01-19',
                '2020-01-20/2020-01-26',
                '2020-01-27/2020-02-02',
                '2020-02-03/2020-02-09',
                '2020-02-10/2020-02-16',
                '2020-02-17/2020-02-23',
                '2020-02-24/2020-03-01',
                '2020-03-02/2020-03-08',
                '2020-03-09/2020-03-15',
                '2020-03-16/2020-03-22',
                '2020-03-23/2020-03-29',
                '2020-03-30/2020-04-05',
                '2020-04-06/2020-04-12',
                '2020-04-13/2020-04-19',
                '2020-04-20/2020-04-26',
                '2020-04-27/2020-05-03',
                '2020-05-04/2020-05-10',
                '2020-05-11/2020-05-17',
                '2020-05-18/2020-05-24',
            ],
            'reconstruction_error': [
                0.795939312162986,
                0.7840110463966236,
                0.8119098730091425,
                0.7982130082187159,
                0.807815521612754,
                0.8492042669464963,
                0.7814127409090083,
                0.8022621626300768,
                0.8104742129966831,
                0.7703901270625767,
                0.8007070128606296,
                0.7953169982962172,
                0.7862784182468701,
                0.838376989270861,
                0.8019280640410021,
                0.7154339372837247,
                0.7171169593894968,
                0.7255999561968017,
                0.73493013255886,
                0.7777717388501538,
            ],
        }
    )
    pd.testing.assert_frame_equal(expected_drift, drift.data[['key', 'reconstruction_error']])


def test_data_reconstruction_drift_calculator_with_only_numeric_should_not_fail(sample_drift_data):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    calc = DataReconstructionDriftCalculator(feature_column_names=['f1', 'f2'], timestamp_column_name='timestamp').fit(
        ref_data
    )
    try:
        calc.calculate(data=sample_drift_data)
    except Exception:
        pytest.fail()


def test_data_reconstruction_drift_calculator_with_only_categorical_should_not_fail(sample_drift_data):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    calc = DataReconstructionDriftCalculator(feature_column_names=['f3', 'f4'], timestamp_column_name='timestamp').fit(
        ref_data
    )
    try:
        calc.calculate(data=sample_drift_data)
    except Exception:
        pytest.fail()


def test_data_reconstruction_drift_calculator_minimum_chunk_size_yields_correct_result(sample_drift_data):  # noqa: D103
    features = ['f1', 'f2', 'f3', 'f4']
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    _ = DataReconstructionDriftCalculator(feature_column_names=['f3', 'f4'], timestamp_column_name='timestamp').fit(
        ref_data
    )
    assert _minimum_chunk_size(features) == 63


def test_data_reconstruction_drift_calculator_given_wrong_cat_imputer_object_raises_typeerror(  # noqa: D103
    sample_drift_data_with_nans,
):
    with pytest.raises(TypeError):
        DataReconstructionDriftCalculator(
            feature_column_names=['f1', 'f2', 'f3', 'f4'],
            timestamp_column_name='timestamp',
            chunk_period='W',
            imputer_categorical=5,
            imputer_continuous=SimpleImputer(missing_values=np.nan, strategy='mean'),
        )


def test_data_reconstruction_drift_calculator_given_wrong_cat_imputer_strategy_raises_valueerror(  # noqa: D103
    sample_drift_data_with_nans,
):
    with pytest.raises(ValueError):
        DataReconstructionDriftCalculator(
            feature_column_names=['f1', 'f2', 'f3', 'f4'],
            timestamp_column_name='timestamp',
            chunk_period='W',
            imputer_categorical=SimpleImputer(missing_values=np.nan, strategy='median'),
            imputer_continuous=SimpleImputer(missing_values=np.nan, strategy='mean'),
        )


def test_data_reconstruction_drift_calculator_given_wrong_cont_imputer_object_raises_typeerror(  # noqa: D103
    sample_drift_data_with_nans,
):
    with pytest.raises(TypeError):
        DataReconstructionDriftCalculator(
            feature_column_names=['f1', 'f2', 'f3', 'f4'],
            timestamp_column_name='timestamp',
            chunk_period='W',
            imputer_categorical=SimpleImputer(missing_values=np.nan, strategy='most_frequent'),
            imputer_continuous=5,
        )


def test_data_reconstruction_drift_calculator_raises_type_error_when_missing_features_column_names(  # noqa: D103
    sample_drift_data,
):
    with pytest.raises(TypeError):
        DataReconstructionDriftCalculator(
            timestamp_column_name='timestamp',
            chunk_period='W',
        )


def test_data_reconstruction_drift_calculator_raises_type_error_when_missing_timestamp_column_name(  # noqa: D103
    sample_drift_data,
):
    with pytest.raises(TypeError):
        DataReconstructionDriftCalculator(
            feature_column_names=['f1', 'f2', 'f3', 'f4'],
            chunk_period='W',
        )
