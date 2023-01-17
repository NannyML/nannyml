#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

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
def sample_drift_data_with_nans(sample_drift_data) -> pd.DataFrame:  # noqa: D103
    data = sample_drift_data.copy(deep=True)
    data['id'] = data.index
    nan_pick1 = set(data.id.sample(frac=0.11, random_state=13))
    nan_pick2 = set(data.id.sample(frac=0.11, random_state=14))
    data.loc[data.id.isin(nan_pick1), 'f1'] = np.NaN
    data.loc[data.id.isin(nan_pick2), 'f4'] = np.NaN
    data.drop(columns=['id'], inplace=True)
    return data


def test_output_drift_calculator_with_params_should_not_fail(sample_drift_data):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    calc = UnivariateDriftCalculator(
        column_names=['output', 'y_pred_proba'],
        timestamp_column_name='timestamp',
        chunk_period='W',
    ).fit(ref_data)
    try:
        _ = calc.calculate(data=sample_drift_data)
    except Exception:
        pytest.fail()


def test_output_drift_calculator_with_default_params_should_not_fail(sample_drift_data):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    calc = UnivariateDriftCalculator(
        column_names=['output', 'y_pred_proba'],
        timestamp_column_name='timestamp',
    ).fit(ref_data)
    try:
        _ = calc.calculate(data=sample_drift_data)
    except Exception:
        pytest.fail()


def test_output_drift_calculator_for_regression_problems():  # noqa: D103
    reference, analysis, _ = load_synthetic_car_price_dataset()
    calc = UnivariateDriftCalculator(
        column_names=['y_pred'],
        timestamp_column_name='timestamp',
        chunk_size=5000,
    ).fit(reference)
    results = calc.calculate(analysis)
    sut = results.filter(period='analysis').to_df()

    assert (
        round(sut['y_pred', 'jensen_shannon', 'value'], 5)
        == [0.02469, 0.03033, 0.02744, 0.03370, 0.03119, 0.02535, 0.29466, 0.30005, 0.30489, 0.29754, 0.30603, 0.30286]
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
                    'y_pred_js': [
                        0.018338079417902442,
                        0.02331144433696094,
                        0.019242884221975648,
                        0.2969888644331771,
                        0.3006797066523378,
                        0.30388591730373204,
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
                    'y_pred_js': [
                        0.018338079417902442,
                        0.02331144433696094,
                        0.019242884221975648,
                        0.2969888644331771,
                        0.3006797066523378,
                        0.30388591730373204,
                    ],
                }
            ),
        ),
        (
            {'chunk_number': 5},
            pd.DataFrame(
                {
                    'key': ['[0:11999]', '[12000:23999]', '[24000:35999]', '[36000:47999]', '[48000:59999]'],
                    'y_pred_js': [
                        0.01792101919275263,
                        0.02262682803313815,
                        0.17026751513773614,
                        0.3002723421679077,
                        0.30312187233953375,
                    ],
                }
            ),
        ),
        (
            {'chunk_number': 5, 'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': ['[0:11999]', '[12000:23999]', '[24000:35999]', '[36000:47999]', '[48000:59999]'],
                    'y_pred_js': [
                        0.01792101919275263,
                        0.02262682803313815,
                        0.17026751513773614,
                        0.3002723421679077,
                        0.30312187233953375,
                    ],
                }
            ),
        ),
        (
            {'chunk_period': 'M', 'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': ['2017-02', '2017-03'],
                    'y_pred_js': [0.028170283827501835, 0.2997831899034714],
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
                    'y_pred_js': [
                        0.02148204835360641,
                        0.028425356316305078,
                        0.027977096337132364,
                        0.02783044431611594,
                        0.02444456834450716,
                        0.29512323724446865,
                        0.2999743688973674,
                        0.30153461857691105,
                        0.3007077108612849,
                        0.3062384802693972,
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
                    'y_pred_js': [
                        0.02148204835360641,
                        0.028425356316305078,
                        0.027977096337132364,
                        0.02783044431611594,
                        0.02444456834450716,
                        0.29512323724446865,
                        0.2999743688973674,
                        0.30153461857691105,
                        0.3007077108612849,
                        0.3062384802693972,
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
def test_univariate_statistical_drift_calculator_for_regression_works_with_chunker(
    calculator_opts, expected  # noqa: D103
):
    reference, analysis, _ = load_synthetic_car_price_dataset()
    calc = UnivariateDriftCalculator(
        column_names=['y_pred'],
        **calculator_opts,
    ).fit(reference)
    results = calc.calculate(analysis)
    sut = results.filter(period='analysis').to_df()[[('chunk', 'chunk', 'key'), ('y_pred', 'jensen_shannon', 'value')]]
    sut.columns = ['key', 'y_pred_js']
    pd.testing.assert_frame_equal(expected, sut)


@pytest.mark.parametrize(
    'calculator_opts, expected',
    [
        (
            {'chunk_size': 10000},
            pd.DataFrame(
                {
                    'key': ['[0:9999]', '[10000:19999]', '[20000:29999]', '[30000:39999]', '[40000:49999]'],
                    'y_pred_js': [
                        0.004366032059329941,
                        0.008204513261276816,
                        0.01201314663099034,
                        0.020600738990536042,
                        0.02289785288459782,
                    ],
                    'y_pred_proba_js': [
                        0.017838328289590907,
                        0.0216236580673744,
                        0.11776323012621548,
                        0.21603251976958054,
                        0.20989914993787034,
                    ],
                }
            ),
        ),
        (
            {'chunk_size': 10000, 'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': ['[0:9999]', '[10000:19999]', '[20000:29999]', '[30000:39999]', '[40000:49999]'],
                    'y_pred_js': [
                        0.004366032059329941,
                        0.008204513261276816,
                        0.01201314663099034,
                        0.020600738990536042,
                        0.02289785288459782,
                    ],
                    'y_pred_proba_js': [
                        0.017838328289590907,
                        0.0216236580673744,
                        0.11776323012621548,
                        0.21603251976958054,
                        0.20989914993787034,
                    ],
                }
            ),
        ),
        (
            {'chunk_number': 5},
            pd.DataFrame(
                {
                    'key': ['[0:9999]', '[10000:19999]', '[20000:29999]', '[30000:39999]', '[40000:49999]'],
                    'y_pred_js': [
                        0.004366032059329941,
                        0.008204513261276816,
                        0.01201314663099034,
                        0.020600738990536042,
                        0.02289785288459782,
                    ],
                    'y_pred_proba_js': [
                        0.017838328289590907,
                        0.0216236580673744,
                        0.11776323012621548,
                        0.21603251976958054,
                        0.20989914993787034,
                    ],
                }
            ),
        ),
        (
            {'chunk_number': 5, 'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': ['[0:9999]', '[10000:19999]', '[20000:29999]', '[30000:39999]', '[40000:49999]'],
                    'y_pred_js': [
                        0.004366032059329941,
                        0.008204513261276816,
                        0.01201314663099034,
                        0.020600738990536042,
                        0.02289785288459782,
                    ],
                    'y_pred_proba_js': [
                        0.017838328289590907,
                        0.0216236580673744,
                        0.11776323012621548,
                        0.21603251976958054,
                        0.20989914993787034,
                    ],
                }
            ),
        ),
        (
            {'chunk_period': 'Y', 'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': ['2017', '2018', '2019', '2020', '2021'],
                    'y_pred_js': [
                        0.017664901069032672,
                        0.008185333575238572,
                        0.014972720080954184,
                        0.02197914784258864,
                        0.08972122347893575,
                    ],
                    'y_pred_proba_js': [
                        0.03018463007070682,
                        0.017165049842576295,
                        0.14821598653099316,
                        0.21422298797745046,
                        0.7103098878216826,
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
                    'y_pred_js': [
                        0.017283823786876047,
                        0.008544249387350317,
                        0.008374381081891114,
                        0.008034645901705951,
                        0.0016478046453817096,
                        0.02238732361810061,
                        0.021366368083352273,
                        0.019835182506606944,
                        0.01235311384948347,
                        0.03345763283697706,
                    ],
                    'y_pred_proba_js': [
                        0.02893294372076363,
                        0.022138893113364276,
                        0.0310428124773422,
                        0.022833280174765977,
                        0.02374738640998019,
                        0.22548550716708837,
                        0.2088146101476805,
                        0.2242815673339107,
                        0.20535156077297334,
                        0.2155386644722171,
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
                    'y_pred_js': [
                        0.017283823786876047,
                        0.008544249387350317,
                        0.008374381081891114,
                        0.008034645901705951,
                        0.0016478046453817096,
                        0.02238732361810061,
                        0.021366368083352273,
                        0.019835182506606944,
                        0.01235311384948347,
                        0.03345763283697706,
                    ],
                    'y_pred_proba_js': [
                        0.02893294372076363,
                        0.022138893113364276,
                        0.0310428124773422,
                        0.022833280174765977,
                        0.02374738640998019,
                        0.22548550716708837,
                        0.2088146101476805,
                        0.2242815673339107,
                        0.20535156077297334,
                        0.2155386644722171,
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
def test_univariate_statistical_drift_calculator_for_binary_classification_works_with_chunker(
    calculator_opts, expected  # noqa: D103
):
    reference, analysis, _ = load_synthetic_binary_classification_dataset()
    reference['y_pred'] = reference['y_pred'].astype("category")
    calc = UnivariateDriftCalculator(
        column_names=['y_pred_proba', 'y_pred'],
        **calculator_opts,
    ).fit(reference)
    results = calc.calculate(analysis)
    sut = results.filter(period='analysis').to_df()[
        [('chunk', 'chunk', 'key'), ('y_pred', 'jensen_shannon', 'value'), ('y_pred_proba', 'jensen_shannon', 'value')]
    ]
    sut.columns = ['key', 'y_pred_js', 'y_pred_proba_js']

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
                    'y_pred_js': [
                        0.006324013882838815,
                        0.004601358156553886,
                        0.01061470567791271,
                        0.06896200120725922,
                        0.07372094889140536,
                        0.06907938357567352,
                    ],
                    'y_pred_proba_upmarket_card_js': [
                        0.01729876493794587,
                        0.017818629872442697,
                        0.02498051961222243,
                        0.2475564938196511,
                        0.24270961682918593,
                        0.24013533794242778,
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
                    'y_pred_js': [
                        0.006324013882838815,
                        0.004601358156553886,
                        0.01061470567791271,
                        0.06896200120725922,
                        0.07372094889140536,
                        0.06907938357567352,
                    ],
                    'y_pred_proba_upmarket_card_js': [
                        0.01729876493794587,
                        0.017818629872442697,
                        0.02498051961222243,
                        0.2475564938196511,
                        0.24270961682918593,
                        0.24013533794242778,
                    ],
                }
            ),
        ),
        (
            {'chunk_number': 5},
            pd.DataFrame(
                {
                    'key': ['[0:11999]', '[12000:23999]', '[24000:35999]', '[36000:47999]', '[48000:59999]'],
                    'y_pred_js': [
                        0.005840552448005313,
                        0.0042187688895671885,
                        0.03624679901959667,
                        0.07365924818646127,
                        0.0702292628381398,
                    ],
                    'y_pred_proba_upmarket_card_js': [
                        0.015334058221434002,
                        0.01626463052861827,
                        0.11865364992201723,
                        0.244427368381236,
                        0.24001077864818152,
                    ],
                }
            ),
        ),
        (
            {'chunk_number': 5, 'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': ['[0:11999]', '[12000:23999]', '[24000:35999]', '[36000:47999]', '[48000:59999]'],
                    'y_pred_js': [
                        0.005840552448005313,
                        0.0042187688895671885,
                        0.03624679901959667,
                        0.07365924818646127,
                        0.0702292628381398,
                    ],
                    'y_pred_proba_upmarket_card_js': [
                        0.015334058221434002,
                        0.01626463052861827,
                        0.11865364992201723,
                        0.244427368381236,
                        0.24001077864818152,
                    ],
                }
            ),
        ),
        (
            {'chunk_period': 'Y', 'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': ['2020', '2021'],
                    'y_pred_js': [0.035365593816879454, 0.05194486561802424],
                    'y_pred_proba_upmarket_card_js': [0.11315897742275421, 0.28776404748812495],
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
                    'y_pred_js': [
                        0.00893645559936086,
                        0.006452416711715143,
                        0.0026477409206083677,
                        0.005881770362799948,
                        0.009799538589281074,
                        0.06566404846491805,
                        0.07073524484367058,
                        0.07662165735865509,
                        0.06702709466498659,
                        0.07349217375412115,
                    ],
                    'y_pred_proba_upmarket_card_js': [
                        0.027071177000638293,
                        0.024023860903177253,
                        0.017546788648652493,
                        0.02826066076992187,
                        0.025546805683291436,
                        0.2476649414014356,
                        0.24354495017280245,
                        0.24616321361387417,
                        0.24265783377877748,
                        0.23900066322004415,
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
                    'y_pred_js': [
                        0.00893645559936086,
                        0.006452416711715143,
                        0.0026477409206083677,
                        0.005881770362799948,
                        0.009799538589281074,
                        0.06566404846491805,
                        0.07073524484367058,
                        0.07662165735865509,
                        0.06702709466498659,
                        0.07349217375412115,
                    ],
                    'y_pred_proba_upmarket_card_js': [
                        0.027071177000638293,
                        0.024023860903177253,
                        0.017546788648652493,
                        0.02826066076992187,
                        0.025546805683291436,
                        0.2476649414014356,
                        0.24354495017280245,
                        0.24616321361387417,
                        0.24265783377877748,
                        0.23900066322004415,
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
def test_univariate_statistical_drift_calculator_for_multiclass_classification_works_with_chunker(
    calculator_opts, expected  # noqa: D103
):
    reference, analysis, _ = load_synthetic_multiclass_classification_dataset()
    calc = UnivariateDriftCalculator(
        column_names=[
            'y_pred',
            'y_pred_proba_upmarket_card',
            'y_pred_proba_highstreet_card',
            'y_pred_proba_prepaid_card',
        ],
        **calculator_opts,
    ).fit(reference)
    results = calc.calculate(analysis)
    sut = results.filter(period='analysis').to_df()[
        [
            ('chunk', 'chunk', 'key'),
            ('y_pred', 'jensen_shannon', 'value'),
            ('y_pred_proba_upmarket_card', 'jensen_shannon', 'value'),
        ]
    ]
    sut.columns = ['key', 'y_pred_js', 'y_pred_proba_upmarket_card_js']

    pd.testing.assert_frame_equal(expected, sut)


@pytest.mark.parametrize(
    'calc_args, plot_args, period',
    [
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'drift'}, 'analysis'),
        ({}, {'kind': 'drift'}, 'analysis'),
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'distribution'}, 'analysis'),
        ({}, {'kind': 'distribution'}, 'analysis'),
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'drift'}, 'all'),
        ({}, {'kind': 'drift'}, 'all'),
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'distribution'}, 'all'),
        ({}, {'kind': 'distribution'}, 'all'),
    ],
    ids=[
        'drift_with_timestamp_without_reference',
        'drift_without_timestamp_without_reference',
        'distribution_with_timestamp_without_reference',
        'distribution_without_timestamp_without_reference',
        'drift_with_timestamp_with_reference',
        'drift_without_timestamp_with_reference',
        'distribution_with_timestamp_with_reference',
        'distribution_without_timestamp_with_reference',
    ],
)
def test_multiclass_classification_result_plots_raise_no_exceptions(calc_args, plot_args, period):  # noqa: D103
    reference, analysis, _ = load_synthetic_multiclass_classification_dataset()
    calc = UnivariateDriftCalculator(
        column_names=[
            'y_pred',
            'y_pred_proba_upmarket_card',
            'y_pred_proba_highstreet_card',
            'y_pred_proba_prepaid_card',
        ],
        **calc_args,
    ).fit(reference)
    sut = calc.calculate(analysis).filter(period=period)

    try:
        _ = sut.plot(**plot_args)
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


@pytest.mark.parametrize(
    'calc_args, plot_args, period',
    [
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'drift'}, 'analysis'),
        ({}, {'kind': 'drift'}, 'analysis'),
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'distribution'}, 'analysis'),
        ({}, {'kind': 'distribution'}, 'analysis'),
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'drift'}, 'all'),
        ({}, {'kind': 'drift'}, 'all'),
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'distribution'}, 'all'),
        ({}, {'kind': 'distribution'}, 'all'),
    ],
    ids=[
        'drift_with_timestamp_without_reference',
        'drift_without_timestamp_without_reference',
        'distribution_with_timestamp_without_reference',
        'distribution_without_timestamp_without_reference',
        'drift_with_timestamp_with_reference',
        'drift_without_timestamp_with_reference',
        'distribution_with_timestamp_with_reference',
        'distribution_without_timestamp_with_reference',
    ],
)
def test_binary_classification_result_plots_raise_no_exceptions(calc_args, plot_args, period):  # noqa: D103
    reference, analysis, _ = load_synthetic_binary_classification_dataset()
    reference['y_pred'] = reference['y_pred'].astype("category")
    calc = UnivariateDriftCalculator(column_names=['y_pred', 'y_pred_proba'], **calc_args).fit(reference)
    sut = calc.calculate(analysis).filter(period=period)

    try:
        _ = sut.plot(**plot_args)
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


@pytest.mark.parametrize(
    'calc_args, plot_args, period',
    [
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'drift'}, 'analysis'),
        ({}, {'kind': 'drift'}, 'analysis'),
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'distribution'}, 'analysis'),
        ({}, {'kind': 'distribution'}, 'analysis'),
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'drift'}, 'all'),
        ({}, {'kind': 'drift'}, 'all'),
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'distribution'}, 'all'),
        ({}, {'kind': 'distribution'}, 'all'),
    ],
    ids=[
        'drift_with_timestamp_without_reference',
        'drift_without_timestamp_without_reference',
        'distribution_with_timestamp_without_reference',
        'distribution_without_timestamp_without_reference',
        'drift_with_timestamp_with_reference',
        'drift_without_timestamp_with_reference',
        'distribution_with_timestamp_with_reference',
        'distribution_without_timestamp_with_reference',
    ],
)
def test_regression_result_plots_raise_no_exceptions(calc_args, plot_args, period):  # noqa: D103
    reference, analysis, _ = load_synthetic_car_price_dataset()
    calc = UnivariateDriftCalculator(column_names=['y_pred'], **calc_args).fit(reference)
    sut = calc.calculate(analysis).filter(period=period)

    try:
        _ = sut.plot(**plot_args)
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")
