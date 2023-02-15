#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from nannyml.datasets import (
    load_synthetic_binary_classification_dataset,
    load_synthetic_car_price_dataset,
    load_synthetic_multiclass_classification_dataset,
)
from nannyml.drift.univariate import UnivariateDriftCalculator


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
def regression_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, tgt_df = load_synthetic_car_price_dataset()

    return ref_df, ana_df, tgt_df


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


def test_target_distribution_calculator_with_params_should_not_fail(sample_drift_data):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    try:
        calc = UnivariateDriftCalculator(
            column_names=['actual'], timestamp_column_name='timestamp', chunk_period='W'
        ).fit(ref_data)
        _ = calc.calculate(data=sample_drift_data)
    except Exception:
        pytest.fail()


def test_target_distribution_calculator_with_default_params_should_not_fail(sample_drift_data):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    try:
        calc = UnivariateDriftCalculator(column_names=['actual']).fit(ref_data)
        _ = calc.calculate(data=sample_drift_data)
    except Exception:
        pytest.fail()


def test_target_distribution_calculator_for_regression_problems_statistical_drift(regression_data):  # noqa: D103
    reference, analysis, analysis_targets = regression_data

    # Get rid of negative values for log based metrics
    reference = regression_data[0][~(regression_data[0]['y_pred'] < 0)]
    analysis = regression_data[1][~(regression_data[1]['y_pred'] < 0)]

    calc = UnivariateDriftCalculator(column_names=['y_true'], timestamp_column_name='timestamp').fit(reference)
    result = calc.calculate(analysis.join(analysis_targets)).filter(period='analysis')
    print(round(result.data[('y_true', 'jensen_shannon', 'value')], 5))
    assert (
        round(result.data[('y_true', 'jensen_shannon', 'value')], 5)
        == [0.02338, 0.02100, 0.02577, 0.03527, 0.03032, 0.24519, 0.25447, 0.25607, 0.25921, 0.25979]
    ).all()


@pytest.mark.parametrize(
    'calculator_opts, expected',
    [
        (
            {'chunk_size': 10000},
            pd.DataFrame(
                {
                    'key': [
                        '[0:9999]',
                        '[10000:19999]',
                        '[20000:29999]',
                        '[30000:39999]',
                        '[40000:49999]',
                        '[50000:59999]',
                    ],
                    'statistical_target_drift': [
                        0.01679512965273928,
                        0.022141971003308808,
                        0.024666539218128822,
                        0.2495460952719345,
                        0.2537200630548825,
                        0.2604930968064078,
                    ],
                }
            ),
        ),
        (
            {'chunk_size': 10000, 'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': [
                        '[0:9999]',
                        '[10000:19999]',
                        '[20000:29999]',
                        '[30000:39999]',
                        '[40000:49999]',
                        '[50000:59999]',
                    ],
                    'statistical_target_drift': [
                        0.01679512965273928,
                        0.022141971003308808,
                        0.024666539218128822,
                        0.2495460952719345,
                        0.2537200630548825,
                        0.2604930968064078,
                    ],
                }
            ),
        ),
        (
            {'chunk_number': 5},
            pd.DataFrame(
                {
                    'key': ['[0:11999]', '[12000:23999]', '[24000:35999]', '[36000:47999]', '[48000:59999]'],
                    'statistical_target_drift': [
                        0.015587328865309872,
                        0.02385711109368465,
                        0.1359047751415131,
                        0.25493613387717934,
                        0.25899939331886124,
                    ],
                }
            ),
        ),
        (
            {'chunk_number': 5, 'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': ['[0:11999]', '[12000:23999]', '[24000:35999]', '[36000:47999]', '[48000:59999]'],
                    'statistical_target_drift': [
                        0.015587328865309872,
                        0.02385711109368465,
                        0.1359047751415131,
                        0.25493613387717934,
                        0.25899939331886124,
                    ],
                }
            ),
        ),
        (
            {'chunk_period': 'M', 'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': ['2017-02', '2017-03'],
                    'statistical_target_drift': [0.025299162121874413, 0.25428924551697557],
                }
            ),
        ),
        (
            {},
            pd.DataFrame(
                {
                    'key': [
                        '[0:5999]',
                        '[6000:11999]',
                        '[12000:17999]',
                        '[18000:23999]',
                        '[24000:29999]',
                        '[30000:35999]',
                        '[36000:41999]',
                        '[42000:47999]',
                        '[48000:53999]',
                        '[54000:59999]',
                    ],
                    'statistical_target_drift': [
                        0.02337819329698633,
                        0.021004795593071005,
                        0.025771666873929577,
                        0.03527178657588183,
                        0.03032476745067173,
                        0.24519412263127713,
                        0.25447077258328393,
                        0.2560733890621882,
                        0.2592139965526459,
                        0.2597909547888234,
                    ],
                }
            ),
        ),
        (
            {'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': [
                        '[0:5999]',
                        '[6000:11999]',
                        '[12000:17999]',
                        '[18000:23999]',
                        '[24000:29999]',
                        '[30000:35999]',
                        '[36000:41999]',
                        '[42000:47999]',
                        '[48000:53999]',
                        '[54000:59999]',
                    ],
                    'statistical_target_drift': [
                        0.02337819329698633,
                        0.021004795593071005,
                        0.025771666873929577,
                        0.03527178657588183,
                        0.03032476745067173,
                        0.24519412263127713,
                        0.25447077258328393,
                        0.2560733890621882,
                        0.2592139965526459,
                        0.2597909547888234,
                    ],
                }
            ),
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
def test_target_drift_for_regression_works_with_chunker(calculator_opts, expected):  # noqa: D103
    reference, analysis, analysis_targets = load_synthetic_car_price_dataset()
    calc = UnivariateDriftCalculator(
        column_names=['y_true'],
        **calculator_opts,
    ).fit(reference)
    results = calc.calculate(analysis.join(analysis_targets))
    sut = results.filter(period='analysis').to_df()[[('chunk', 'chunk', 'key'), ('y_true', 'jensen_shannon', 'value')]]
    sut.columns = ['key', 'statistical_target_drift']
    pd.testing.assert_frame_equal(expected, sut)


@pytest.mark.parametrize(
    'calculator_opts, expected',
    [
        (
            {'chunk_size': 10000},
            pd.DataFrame(
                {
                    'key': ['[0:9999]', '[10000:19999]', '[20000:29999]', '[30000:39999]', '[40000:49999]'],
                    'statistical_target_drift': [
                        0.004093771427313285,
                        0.007202604870580528,
                        0.0012060373619027277,
                        0.002734826305568785,
                        0.0038389670681924274,
                    ],
                }
            ),
        ),
        (
            {'chunk_size': 10000, 'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': ['[0:9999]', '[10000:19999]', '[20000:29999]', '[30000:39999]', '[40000:49999]'],
                    'statistical_target_drift': [
                        0.004093771427313285,
                        0.007202604870580528,
                        0.0012060373619027277,
                        0.002734826305568785,
                        0.0038389670681924274,
                    ],
                }
            ),
        ),
        (
            {'chunk_number': 5},
            pd.DataFrame(
                {
                    'key': ['[0:9999]', '[10000:19999]', '[20000:29999]', '[30000:39999]', '[40000:49999]'],
                    'statistical_target_drift': [
                        0.004093771427313285,
                        0.007202604870580528,
                        0.0012060373619027277,
                        0.002734826305568785,
                        0.0038389670681924274,
                    ],
                }
            ),
        ),
        (
            {'chunk_number': 5, 'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': ['[0:9999]', '[10000:19999]', '[20000:29999]', '[30000:39999]', '[40000:49999]'],
                    'statistical_target_drift': [
                        0.004093771427313285,
                        0.007202604870580528,
                        0.0012060373619027277,
                        0.002734826305568785,
                        0.0038389670681924274,
                    ],
                }
            ),
        ),
        (
            {'chunk_period': 'Y', 'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': ['2017', '2018', '2019', '2020', '2021'],
                    'statistical_target_drift': [
                        0.015280820030146264,
                        0.006911912044357547,
                        0.0021698147637991376,
                        0.0029091172643770785,
                        0.08507929435324538,
                    ],
                }
            ),
        ),
        (
            {},
            pd.DataFrame(
                {
                    'key': [
                        '[0:4999]',
                        '[5000:9999]',
                        '[10000:14999]',
                        '[15000:19999]',
                        '[20000:24999]',
                        '[25000:29999]',
                        '[30000:34999]',
                        '[35000:39999]',
                        '[40000:44999]',
                        '[45000:49999]',
                    ],
                    'statistical_target_drift': [
                        0.014967545228197255,
                        0.006777887586933271,
                        0.011705090833295635,
                        0.002700867424723671,
                        0.002564959881244088,
                        0.00015287800412266007,
                        0.004603383709406135,
                        0.0008663083254600605,
                        0.002700867424723671,
                        0.010379513761252155,
                    ],
                }
            ),
        ),
        (
            {'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': [
                        '[0:4999]',
                        '[5000:9999]',
                        '[10000:14999]',
                        '[15000:19999]',
                        '[20000:24999]',
                        '[25000:29999]',
                        '[30000:34999]',
                        '[35000:39999]',
                        '[40000:44999]',
                        '[45000:49999]',
                    ],
                    'statistical_target_drift': [
                        0.014967545228197255,
                        0.006777887586933271,
                        0.011705090833295635,
                        0.002700867424723671,
                        0.002564959881244088,
                        0.00015287800412266007,
                        0.004603383709406135,
                        0.0008663083254600605,
                        0.002700867424723671,
                        0.010379513761252155,
                    ],
                }
            ),
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
def test_target_drift_for_binary_classification_works_with_chunker(calculator_opts, expected):  # noqa: D103
    reference, analysis, analysis_targets = load_synthetic_binary_classification_dataset()
    reference['work_home_actual'] = reference['work_home_actual'].astype('category')
    analysis_targets['work_home_actual'] = analysis_targets['work_home_actual'].astype('category')
    calc = UnivariateDriftCalculator(
        column_names=['work_home_actual'],
        **calculator_opts,
    ).fit(reference)
    results = calc.calculate(analysis.merge(analysis_targets, on='identifier'))
    sut = results.filter(period='analysis').to_df()[
        [('chunk', 'chunk', 'key'), ('work_home_actual', 'jensen_shannon', 'value')]
    ]
    sut.columns = ['key', 'statistical_target_drift']
    pd.testing.assert_frame_equal(expected, sut)


@pytest.mark.parametrize(
    'calculator_opts, expected',
    [
        (
            {'chunk_size': 10000},
            pd.DataFrame(
                {
                    'key': [
                        '[0:9999]',
                        '[10000:19999]',
                        '[20000:29999]',
                        '[30000:39999]',
                        '[40000:49999]',
                        '[50000:59999]',
                    ],
                    'statistical_target_drift': [
                        0.0016425066364221604,
                        0.008220692714373961,
                        0.005229199578465021,
                        0.020204229160376244,
                        0.019581827343873603,
                        0.022630272418250677,
                    ],
                }
            ),
        ),
        (
            {'chunk_size': 10000, 'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': [
                        '[0:9999]',
                        '[10000:19999]',
                        '[20000:29999]',
                        '[30000:39999]',
                        '[40000:49999]',
                        '[50000:59999]',
                    ],
                    'statistical_target_drift': [
                        0.0016425066364221604,
                        0.008220692714373961,
                        0.005229199578465021,
                        0.020204229160376244,
                        0.019581827343873603,
                        0.022630272418250677,
                    ],
                }
            ),
        ),
        (
            {'chunk_number': 5},
            pd.DataFrame(
                {
                    'key': ['[0:11999]', '[12000:23999]', '[24000:35999]', '[36000:47999]', '[48000:59999]'],
                    'statistical_target_drift': [
                        0.0021929356780220417,
                        0.0064669042170913986,
                        0.015029368002268359,
                        0.019796501732703135,
                        0.021265582906514667,
                    ],
                }
            ),
        ),
        (
            {'chunk_number': 5, 'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': ['[0:11999]', '[12000:23999]', '[24000:35999]', '[36000:47999]', '[48000:59999]'],
                    'statistical_target_drift': [
                        0.0021929356780220417,
                        0.0064669042170913986,
                        0.015029368002268359,
                        0.019796501732703135,
                        0.021265582906514667,
                    ],
                }
            ),
        ),
        (
            {'chunk_period': 'Y', 'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {'key': ['2020', '2021'], 'statistical_target_drift': [0.010759094317127073, 0.01341938637844519]}
            ),
        ),
        (
            {},
            pd.DataFrame(
                {
                    'key': [
                        '[0:5999]',
                        '[6000:11999]',
                        '[12000:17999]',
                        '[18000:23999]',
                        '[24000:29999]',
                        '[30000:35999]',
                        '[36000:41999]',
                        '[42000:47999]',
                        '[48000:53999]',
                        '[54000:59999]',
                    ],
                    'statistical_target_drift': [
                        0.004146764466577644,
                        0.008375672970699688,
                        0.00558342874507466,
                        0.008389613994787881,
                        0.008580819685195912,
                        0.022006258877631527,
                        0.014131730114252177,
                        0.02578136410091765,
                        0.021744805155907727,
                        0.020791075713502798,
                    ],
                }
            ),
        ),
        (
            {'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': [
                        '[0:5999]',
                        '[6000:11999]',
                        '[12000:17999]',
                        '[18000:23999]',
                        '[24000:29999]',
                        '[30000:35999]',
                        '[36000:41999]',
                        '[42000:47999]',
                        '[48000:53999]',
                        '[54000:59999]',
                    ],
                    'statistical_target_drift': [
                        0.004146764466577644,
                        0.008375672970699688,
                        0.00558342874507466,
                        0.008389613994787881,
                        0.008580819685195912,
                        0.022006258877631527,
                        0.014131730114252177,
                        0.02578136410091765,
                        0.021744805155907727,
                        0.020791075713502798,
                    ],
                }
            ),
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
def test_target_drift_for_multiclass_classification_works_with_chunker(calculator_opts, expected):  # noqa: D103
    reference, analysis, analysis_targets = load_synthetic_multiclass_classification_dataset()
    calc = UnivariateDriftCalculator(
        column_names=['y_true'],
        **calculator_opts,
    ).fit(reference)
    results = calc.calculate(analysis.merge(analysis_targets, left_index=True, right_index=True)).filter(
        period='analysis'
    )
    sut = results.filter(period='analysis').to_df()[[('chunk', 'chunk', 'key'), ('y_true', 'jensen_shannon', 'value')]]
    sut.columns = ['key', 'statistical_target_drift']
    pd.testing.assert_frame_equal(expected, sut)


@pytest.mark.parametrize(
    'calc_args, plot_args, period',
    [
        (
            {'timestamp_column_name': 'timestamp'},
            {'kind': 'drift', 'column_name': 'y_true', 'method': 'jensen_shannon'},
            'analysis',
        ),
        ({}, {'kind': 'drift', 'column_name': 'y_true', 'method': 'jensen_shannon'}, 'analysis'),
        (
            {'timestamp_column_name': 'timestamp'},
            {'kind': 'drift', 'column_name': 'y_true', 'method': 'jensen_shannon'},
            'all',
        ),
        ({}, {'kind': 'drift', 'column_name': 'y_true', 'method': 'jensen_shannon'}, 'all'),
        (
            {'timestamp_column_name': 'timestamp'},
            {'kind': 'distribution', 'column_name': 'y_true', 'method': 'jensen_shannon'},
            'analysis',
        ),
        ({}, {'kind': 'distribution', 'column_name': 'y_true', 'method': 'jensen_shannon'}, 'analysis'),
        (
            {'timestamp_column_name': 'timestamp'},
            {'kind': 'distribution', 'column_name': 'y_true', 'method': 'jensen_shannon'},
            'all',
        ),
        ({}, {'kind': 'distribution', 'column_name': 'y_true', 'method': 'jensen_shannon'}, 'all'),
    ],
    ids=[
        'target_drift_with_timestamp_without_reference',
        'target_drift_without_timestamp_without_reference',
        'target_drift_with_timestamp_with_reference',
        'target_drift_without_timestamp_with_reference',
        'target_distribution_with_timestamp_without_reference',
        'target_distribution_without_timestamp_without_reference',
        'target_distribution_with_timestamp_with_reference',
        'target_distribution_without_timestamp_with_reference',
    ],
)
def test_multiclass_classification_result_plots_raise_no_exceptions(calc_args, plot_args, period):  # noqa: D103
    reference, analysis, analysis_targets = load_synthetic_multiclass_classification_dataset()
    calc = UnivariateDriftCalculator(column_names=['y_true'], **calc_args).fit(reference)
    sut = calc.calculate(analysis.merge(analysis_targets, left_index=True, right_index=True)).filter(period=period)

    try:
        _ = sut.plot(**plot_args)
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


@pytest.mark.parametrize(
    'calc_args, plot_args, period',
    [
        (
            {'timestamp_column_name': 'timestamp'},
            {'kind': 'drift', 'column_name': 'work_home_actual', 'method': 'jensen_shannon'},
            'analysis',
        ),
        ({}, {'kind': 'drift', 'column_name': 'work_home_actual', 'method': 'jensen_shannon'}, 'analysis'),
        (
            {'timestamp_column_name': 'timestamp'},
            {'kind': 'drift', 'column_name': 'work_home_actual', 'method': 'jensen_shannon'},
            'all',
        ),
        ({}, {'kind': 'drift', 'column_name': 'work_home_actual', 'method': 'jensen_shannon'}, 'all'),
        (
            {'timestamp_column_name': 'timestamp'},
            {
                'kind': 'distribution',
                'column_name': 'work_home_actual',
                'method': 'jensen_shannon',
            },
            'analysis',
        ),
        (
            {},
            {
                'kind': 'distribution',
                'column_name': 'work_home_actual',
                'method': 'jensen_shannon',
            },
            'analysis',
        ),
        (
            {'timestamp_column_name': 'timestamp'},
            {
                'kind': 'distribution',
                'column_name': 'work_home_actual',
                'method': 'jensen_shannon',
            },
            'all',
        ),
        (
            {},
            {
                'kind': 'distribution',
                'column_name': 'work_home_actual',
                'method': 'jensen_shannon',
            },
            'all',
        ),
    ],
    ids=[
        'target_drift_with_timestamp_without_reference',
        'target_drift_without_timestamp_without_reference',
        'target_drift_with_timestamp_with_reference',
        'target_drift_without_timestamp_with_reference',
        'target_distribution_with_timestamp_without_reference',
        'target_distribution_without_timestamp_without_reference',
        'target_distribution_with_timestamp_with_reference',
        'target_distribution_without_timestamp_with_reference',
    ],
)
def test_binary_classification_result_plots_raise_no_exceptions(calc_args, plot_args, period):  # noqa: D103
    reference, analysis, analysis_targets = load_synthetic_binary_classification_dataset()
    reference['work_home_actual'] = reference['work_home_actual'].astype('category')
    analysis_targets['work_home_actual'] = analysis_targets['work_home_actual'].astype('category')
    calc = UnivariateDriftCalculator(column_names=['work_home_actual'], **calc_args).fit(reference)
    sut = calc.calculate(analysis.merge(analysis_targets, on='identifier')).filter(period=period)

    try:
        _ = sut.plot(**plot_args)
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


@pytest.mark.parametrize(
    'calc_args, plot_args, period',
    [
        (
            {'timestamp_column_name': 'timestamp'},
            {'kind': 'drift', 'column_name': 'y_true', 'method': 'jensen_shannon'},
            'analysis',
        ),
        ({}, {'kind': 'drift', 'column_name': 'y_true', 'method': 'jensen_shannon'}, 'analysis'),
        (
            {'timestamp_column_name': 'timestamp'},
            {'kind': 'drift', 'column_name': 'y_true', 'method': 'jensen_shannon'},
            'all',
        ),
        ({}, {'kind': 'drift', 'column_name': 'y_true', 'method': 'jensen_shannon'}, 'all'),
        (
            {'timestamp_column_name': 'timestamp'},
            {'kind': 'distribution', 'column_name': 'y_true', 'method': 'jensen_shannon'},
            'analysis',
        ),
        ({}, {'kind': 'distribution', 'column_name': 'y_true', 'method': 'jensen_shannon'}, 'analysis'),
        (
            {'timestamp_column_name': 'timestamp'},
            {'kind': 'distribution', 'column_name': 'y_true', 'method': 'jensen_shannon'},
            'all',
        ),
        ({}, {'kind': 'distribution', 'column_name': 'y_true', 'method': 'jensen_shannon'}, 'all'),
    ],
    ids=[
        'target_drift_with_timestamp_without_reference',
        'target_drift_without_timestamp_without_reference',
        'target_drift_with_timestamp_with_reference',
        'target_drift_without_timestamp_with_reference',
        'target_distribution_with_timestamp_without_reference',
        'target_distribution_without_timestamp_without_reference',
        'target_distribution_with_timestamp_with_reference',
        'target_distribution_without_timestamp_with_reference',
    ],
)
def test_regression_result_plots_raise_no_exceptions(calc_args, plot_args, period):  # noqa: D103
    reference, analysis, analysis_targets = load_synthetic_car_price_dataset()
    calc = UnivariateDriftCalculator(column_names=['y_true'], **calc_args).fit(reference)
    sut = calc.calculate(analysis.join(analysis_targets)).filter(period=period)

    try:
        _ = sut.plot(**plot_args)
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")
