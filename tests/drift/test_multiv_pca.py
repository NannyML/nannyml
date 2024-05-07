#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Tests for Multivariate Data Reconstruction package."""

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from nannyml._typing import Result
from nannyml.chunk import PeriodBasedChunker, SizeBasedChunker
from nannyml.drift.multivariate.data_reconstruction.calculator import DataReconstructionDriftCalculator
from nannyml.drift.univariate import UnivariateDriftCalculator
from nannyml.performance_estimation.confidence_based import CBPE


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
def reconstruction_drift_result(sample_drift_data) -> Result:
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    calc = DataReconstructionDriftCalculator(
        column_names=['f1', 'f2', 'f3', 'f4'],
        timestamp_column_name='timestamp',
        n_components=0.75,
        chunk_period='W',
    ).fit(ref_data)
    return calc.calculate(data=sample_drift_data)


def test_data_reconstruction_drift_calculator_with_params_should_not_fail(sample_drift_data):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    calc = DataReconstructionDriftCalculator(
        column_names=['f1', 'f2', 'f3', 'f4'],
        timestamp_column_name='timestamp',
        n_components=0.75,
        chunk_period='W',
    ).fit(ref_data)
    try:
        _ = calc.calculate(data=sample_drift_data)
    except Exception:
        pytest.fail()


def test_data_reconstruction_drift_calculator_with_default_params_should_not_fail(sample_drift_data):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    calc = DataReconstructionDriftCalculator(
        column_names=['f1', 'f2', 'f3', 'f4'], timestamp_column_name='timestamp'
    ).fit(ref_data)
    try:
        _ = calc.calculate(data=sample_drift_data)
    except Exception:
        pytest.fail()


def test_data_reconstruction_drift_calculator_with_default_params_should_not_fail_w_nans(  # noqa: D103
    sample_drift_data_with_nans,
):
    ref_data = sample_drift_data_with_nans.loc[sample_drift_data_with_nans['period'] == 'reference']
    calc = DataReconstructionDriftCalculator(
        column_names=['f1', 'f2', 'f3', 'f4'],
        timestamp_column_name='timestamp',
        n_components=0.75,
        chunk_period='W',
    ).fit(ref_data)
    try:
        _ = calc.calculate(data=sample_drift_data_with_nans)
    except Exception:
        pytest.fail()


def test_data_reconstruction_drift_calculator_should_contain_chunk_details_and_single_drift_value_column(  # noqa: D103
    sample_drift_data,
):
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    calc = DataReconstructionDriftCalculator(
        column_names=['f1', 'f2', 'f3', 'f4'],
        timestamp_column_name='timestamp',
        n_components=0.75,
        chunk_period='W',
    ).fit(ref_data)
    drift = calc.calculate(data=sample_drift_data)

    sut = drift.data.columns
    assert len(sut) == 14
    assert ('chunk', 'key') in sut
    assert ('chunk', 'chunk_index') in sut
    assert ('chunk', 'start_index') in sut
    assert ('chunk', 'start_date') in sut
    assert ('chunk', 'end_index') in sut
    assert ('chunk', 'end_date') in sut
    assert ('chunk', 'period') in sut
    assert ('reconstruction_error', 'sampling_error') in sut
    assert ('reconstruction_error', 'upper_confidence_boundary') in sut
    assert ('reconstruction_error', 'lower_confidence_boundary') in sut
    assert ('reconstruction_error', 'upper_threshold') in sut
    assert ('reconstruction_error', 'lower_threshold') in sut
    assert ('reconstruction_error', 'alert') in sut
    assert ('reconstruction_error', 'value') in sut


def test_data_reconstruction_drift_calculator_should_contain_a_row_for_each_chunk(sample_drift_data):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    calc = DataReconstructionDriftCalculator(
        column_names=['f1', 'f2', 'f3', 'f4'],
        timestamp_column_name='timestamp',
        n_components=0.75,
        chunk_period='W',
    ).fit(ref_data)
    drift = calc.calculate(data=sample_drift_data).filter(period='analysis')

    expected = len(PeriodBasedChunker(offset='W', timestamp_column_name='timestamp').split(sample_drift_data))
    sut = len(drift.data)
    assert sut == expected


# TODO: find a better way to test this
def test_data_reconstruction_drift_calculator_should_not_fail_when_using_feature_subset(  # noqa: D103
    sample_drift_data,
):
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']

    calc = DataReconstructionDriftCalculator(column_names=['f1', 'f3'], timestamp_column_name='timestamp').fit(ref_data)
    try:
        calc.fit(ref_data)
        calc.calculate(sample_drift_data)
    except Exception as exc:
        pytest.fail(f"should not have failed but got {exc}")


def test_data_reconstruction_drift_calculator_numeric_results(sample_drift_data):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']

    # calc = DataReconstructionDriftCalculator(sample_drift_metadata, chunk_period='W').fit(ref_data)
    calc = DataReconstructionDriftCalculator(
        column_names=['f1', 'f2', 'f3', 'f4'], timestamp_column_name='timestamp', chunk_period='W'
    ).fit(ref_data)
    drift = calc.calculate(data=sample_drift_data).filter(period='analysis')
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
    sut = drift.to_df()[[('chunk', 'key'), ('reconstruction_error', 'value')]]
    sut.columns = ['key', 'reconstruction_error']
    pd.testing.assert_frame_equal(expected_drift, sut)


@pytest.mark.parametrize(
    'calculator_opts, expected',
    [
        (
            {'chunk_size': 5000},
            [0.79987, 0.80210, 0.80430, 0.73552, 0.76087],
        ),
        (
            {'chunk_size': 5000, 'timestamp_column_name': 'timestamp'},
            [0.79987, 0.80210, 0.80430, 0.73552, 0.76087],
        ),
        (
            {'chunk_number': 5},
            [0.7975183099468669, 0.8101736730245841, 0.7942220878040264, 0.7855043522106143, 0.7388546967488279],
        ),
        (
            {'chunk_number': 5, 'timestamp_column_name': 'timestamp'},
            [0.7975183099468669, 0.8101736730245841, 0.7942220878040264, 0.7855043522106143, 0.7388546967488279],
        ),
        (
            {'chunk_period': 'M', 'timestamp_column_name': 'timestamp'},
            [0.7925562396242019, 0.81495562506899, 0.7914354678003803, 0.7766351972000973, 0.7442465240638783],
        ),
        (
            {},
            [
                0.7899751792798048,
                0.805061440613929,
                0.828509894279626,
                0.7918374517695422,
                0.7904321700296298,
                0.798012005578423,
                0.8123277037588652,
                0.7586810006623634,
                0.721358457793149,
                0.7563509357045066,
            ],
        ),
        (
            {'timestamp_column_name': 'timestamp'},
            [
                0.7899751792798048,
                0.805061440613929,
                0.828509894279626,
                0.7918374517695422,
                0.7904321700296298,
                0.798012005578423,
                0.8123277037588652,
                0.7586810006623634,
                0.721358457793149,
                0.7563509357045066,
            ],
        ),
    ],
    ids=[
        'size_based_without_timestamp',
        'size_based_with_timestamp',
        'count_based_without_timestamp',
        'count_based_with_timestamp',
        'period_based_with_timestamp',
        'default_without_timestamp',
        'default_with_timestamp',
    ],
)
def test_data_reconstruction_drift_calculator_works_with_chunker(
    sample_drift_data, calculator_opts, expected  # noqa: D103
):
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    calc = DataReconstructionDriftCalculator(column_names=['f1', 'f2', 'f3', 'f4'], **calculator_opts).fit(ref_data)
    sut = calc.calculate(data=sample_drift_data).filter(period='analysis').to_df()

    assert all(round(sut.loc[:, ('reconstruction_error', 'value')], 5) == [round(n, 5) for n in expected])


def test_data_reconstruction_drift_calculator_with_only_numeric_should_not_fail(sample_drift_data):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    calc = DataReconstructionDriftCalculator(column_names=['f1', 'f2'], timestamp_column_name='timestamp').fit(ref_data)
    try:
        calc.calculate(data=sample_drift_data)
    except Exception:
        pytest.fail()


def test_data_reconstruction_drift_calculator_with_only_categorical_should_not_fail(sample_drift_data):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    calc = DataReconstructionDriftCalculator(column_names=['f3', 'f4'], timestamp_column_name='timestamp').fit(ref_data)
    try:
        calc.calculate(data=sample_drift_data)
    except Exception:
        pytest.fail()


def test_data_reconstruction_drift_calculator_given_wrong_cat_imputer_object_raises_typeerror(  # noqa: D103
    sample_drift_data_with_nans,
):
    with pytest.raises(TypeError):
        DataReconstructionDriftCalculator(
            column_names=['f1', 'f2', 'f3', 'f4'],
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
            column_names=['f1', 'f2', 'f3', 'f4'],
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
            column_names=['f1', 'f2', 'f3', 'f4'],
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


# See https://github.com/NannyML/nannyml/issues/179
def test_data_reconstruction_drift_lower_threshold_smaller_than_upper_threshold(sample_drift_data):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']

    calc = DataReconstructionDriftCalculator(
        column_names=['f1', 'f2', 'f3', 'f4'], timestamp_column_name='timestamp'
    ).fit(ref_data)
    results = calc.calculate(data=sample_drift_data)

    results_df = results.to_df()
    assert all(
        results_df.loc[:, ('reconstruction_error', 'lower_threshold')]
        <= results_df.loc[:, ('reconstruction_error', 'upper_threshold')]
    )


# See https://github.com/NannyML/nannyml/issues/192
def test_data_reconstruction_drift_calculator_returns_distinct_but_consistent_results_when_reused(sample_drift_data):
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    sut = DataReconstructionDriftCalculator(column_names=['f1', 'f2', 'f3', 'f4']).fit(ref_data)
    result1 = sut.calculate(data=sample_drift_data)
    result2 = sut.calculate(data=sample_drift_data)

    assert result1 is not result2
    pd.testing.assert_frame_equal(result1.to_df(), result2.to_df())


# See https://github.com/NannyML/nannyml/issues/197
def test_data_reconstruction_drift_result_filter_should_preserve_data_with_default_args(reconstruction_drift_result):
    filtered_result = reconstruction_drift_result.filter()
    assert filtered_result.data.equals(reconstruction_drift_result.data)


# See https://github.com/NannyML/nannyml/issues/197
def test_data_reconstruction_drift_result_filter_period(reconstruction_drift_result):
    ref_period = reconstruction_drift_result.data.loc[
        reconstruction_drift_result.data.loc[:, ("chunk", "period")] == "reference", :
    ]
    filtered_result = reconstruction_drift_result.filter(period="reference")
    assert filtered_result.data.equals(ref_period)


def test_data_reconstruction_drift_chunked_by_period_has_variable_sampling_error(sample_drift_data):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']

    calc = DataReconstructionDriftCalculator(
        column_names=['f1', 'f2', 'f3', 'f4'], timestamp_column_name='timestamp', chunk_period='M'
    ).fit(ref_data)
    results = calc.calculate(data=sample_drift_data)

    assert ('reconstruction_error', 'sampling_error') in results.data.columns
    assert np.array_equal(
        np.round(results.filter(period='analysis').to_df().loc[:, ('reconstruction_error', 'sampling_error')], 4),
        [0.0095, 0.0090, 0.0086, 0.0086, 0.0092],
    )


@pytest.mark.parametrize(
    'calc_args, plot_args, period',
    [
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'drift'}, 'analysis'),
        ({}, {'kind': 'drift'}, 'analysis'),
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'drift'}, 'all'),
        ({}, {'kind': 'drift'}, 'all'),
    ],
    ids=[
        'drift_with_timestamp_without_reference',
        'drift_without_timestamp_without_reference',
        'drift_with_timestamp_with_reference',
        'drift_without_timestamp_with_reference',
    ],
)
def test_result_plots_raise_no_exceptions(sample_drift_data, calc_args, plot_args, period):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    ana_data = sample_drift_data.loc[sample_drift_data['period'] == 'analysis']

    calc = DataReconstructionDriftCalculator(column_names=['f1', 'f2', 'f3', 'f4'], **calc_args).fit(ref_data)
    sut = calc.calculate(data=ana_data).filter(period=period)

    try:
        _ = sut.plot(**plot_args)
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


def test_result_comparison_to_univariate_drift_plots_raise_no_exceptions(sample_drift_data):
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    ana_data = sample_drift_data.loc[sample_drift_data['period'] == 'analysis']

    calc = DataReconstructionDriftCalculator(column_names=['f1', 'f2', 'f3', 'f4']).fit(ref_data)
    result = calc.calculate(ana_data)

    calc2 = UnivariateDriftCalculator(
        column_names=['f1', 'f3'],
        continuous_methods=['kolmogorov_smirnov'],
        categorical_methods=['chi2'],
        timestamp_column_name='timestamp',
    ).fit(ref_data)
    result2 = calc2.calculate(ana_data)

    try:
        _ = result.compare(result2).plot()
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


def test_result_comparison_to_cbpe_plots_raise_no_exceptions(sample_drift_data):
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    ana_data = sample_drift_data.loc[sample_drift_data['period'] == 'analysis']

    calc = DataReconstructionDriftCalculator(column_names=['f1', 'f2', 'f3', 'f4']).fit(ref_data)
    result = calc.calculate(ana_data)

    calc2 = CBPE(
        timestamp_column_name='timestamp',
        y_pred_proba='y_pred_proba',
        y_pred='output',
        y_true='actual',
        metrics=['roc_auc', 'f1'],
        problem_type='classification_binary',
    ).fit(ref_data)
    result2 = calc2.estimate(ana_data)

    try:
        _ = result.compare(result2).plot()
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


def test_data_reconstruction_drift_chunked_by_size(sample_drift_data):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']

    chunker = SizeBasedChunker(chunk_size=2500, incomplete='drop')

    calc = DataReconstructionDriftCalculator(
        column_names=['f1', 'f2', 'f3', 'f4'], timestamp_column_name='timestamp', chunker=chunker
    ).fit(ref_data)
    results = calc.calculate(data=sample_drift_data.head(7500))

    assert ('reconstruction_error', 'sampling_error') in results.data.columns
    assert np.array_equal(
        np.round(
            results.filter(period='analysis').to_df().loc[:, ('reconstruction_error', 'sampling_error')],
            4),
        [0.0118, 0.0115, 0.0118],
    )
