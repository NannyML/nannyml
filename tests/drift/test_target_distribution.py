#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from nannyml._typing import ProblemType
from nannyml.datasets import (
    load_synthetic_binary_classification_dataset,
    load_synthetic_car_price_dataset,
    load_synthetic_multiclass_classification_dataset,
)
from nannyml.drift.target.target_distribution import TargetDistributionCalculator


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
    calc = TargetDistributionCalculator(
        y_true='actual', timestamp_column_name='timestamp', chunk_period='W', problem_type='classification_binary'
    ).fit(ref_data)
    try:
        _ = calc.calculate(data=sample_drift_data)
    except Exception:
        pytest.fail()


def test_target_distribution_calculator_with_default_params_should_not_fail(sample_drift_data):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    calc = TargetDistributionCalculator(
        y_true='actual', timestamp_column_name='timestamp', problem_type='classification_binary'
    ).fit(ref_data)
    try:
        _ = calc.calculate(data=sample_drift_data)
    except Exception:
        pytest.fail()


def test_target_distribution_calculator_for_regression_problems_statistical_drift(regression_data):  # noqa: D103
    reference, analysis, analysis_targets = regression_data

    # Get rid of negative values for log based metrics
    reference = regression_data[0][~(regression_data[0]['y_pred'] < 0)]
    analysis = regression_data[1][~(regression_data[1]['y_pred'] < 0)]

    calc = TargetDistributionCalculator(
        y_true='y_true', timestamp_column_name='timestamp', problem_type='regression'
    ).fit(reference)
    result = calc.calculate(analysis.join(analysis_targets)).filter(period='analysis')

    assert (
        round(result.data['statistical_target_drift'], 5)
        == [0.01425, 0.01657, 0.01007, 0.01192, 0.00867, 0.17168, 0.18012, 0.17907, 0.18323, 0.18738]
    ).all()


def test_target_distribution_calculator_for_regression_problems_mean_drift(regression_data):  # noqa: D103
    reference, analysis, analysis_targets = regression_data

    # Get rid of negative values for log based metrics
    reference = regression_data[0][~(regression_data[0]['y_pred'] < 0)]
    analysis = regression_data[1][~(regression_data[1]['y_pred'] < 0)]

    calc = TargetDistributionCalculator(
        y_true='y_true', timestamp_column_name='timestamp', problem_type='regression'
    ).fit(reference)
    result = calc.calculate(analysis.join(analysis_targets))
    sut = result.data[result.data['period'] == 'analysis'].reset_index(drop=True)

    assert (
        round(sut['metric_target_drift'], 5)
        == [
            4862.94117,
            4790.5815,
            4793.34933,
            4838.25617,
            4799.1335,
            4852.63667,
            4875.45667,
            4867.589,
            4885.108,
            4787.09417,
        ]
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
                    'metric_target_drift': [4834.7893, 4776.1287, 4839.639, 4868.6151, 4873.9113, 4818.2043],
                    'statistical_target_drift': [
                        0.014583333333333337,
                        0.006916666666666682,
                        0.008950000000000014,
                        0.17545,
                        0.178,
                        0.1867666666666667,
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
                    'metric_target_drift': [4834.7893, 4776.1287, 4839.639, 4868.6151, 4873.9113, 4818.2043],
                    'statistical_target_drift': [
                        0.014583333333333337,
                        0.006916666666666682,
                        0.008950000000000014,
                        0.17545,
                        0.178,
                        0.1867666666666667,
                    ],
                }
            ),
        ),
        (
            {'chunk_number': 5},
            pd.DataFrame(
                {
                    'key': ['[0:11999]', '[12000:23999]', '[24000:35999]', '[36000:47999]', '[48000:59999]'],
                    'metric_target_drift': [
                        4826.761333333333,
                        4815.80275,
                        4825.885083333334,
                        4871.522833333333,
                        4836.101083333333,
                    ],
                    'statistical_target_drift': [
                        0.014516666666666678,
                        0.00869999999999993,
                        0.08675,
                        0.1785,
                        0.18508333333333332,
                    ],
                }
            ),
        ),
        (
            {'chunk_number': 5, 'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': ['[0:11999]', '[12000:23999]', '[24000:35999]', '[36000:47999]', '[48000:59999]'],
                    'metric_target_drift': [
                        4826.761333333333,
                        4815.80275,
                        4825.885083333334,
                        4871.522833333333,
                        4836.101083333333,
                    ],
                    'statistical_target_drift': [
                        0.014516666666666678,
                        0.00869999999999993,
                        0.08675,
                        0.1785,
                        0.18508333333333332,
                    ],
                }
            ),
        ),
        (
            {'chunk_period': 'M', 'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': ['2017-02', '2017-03'],
                    'metric_target_drift': [4826.1297808607915, 4845.4011313417],
                    'statistical_target_drift': [0.008650415155814883, 0.17979488539272875],
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
                    'metric_target_drift': [
                        4862.9411666666665,
                        4790.5815,
                        4793.349333333334,
                        4838.256166666667,
                        4799.1335,
                        4852.636666666666,
                        4875.456666666667,
                        4867.589,
                        4885.108,
                        4787.094166666667,
                    ],
                    'statistical_target_drift': [
                        0.014249999999999985,
                        0.016566666666666674,
                        0.010066666666666668,
                        0.011916666666666659,
                        0.008666666666666656,
                        0.1716833333333333,
                        0.18011666666666665,
                        0.17906666666666665,
                        0.18323333333333333,
                        0.1873833333333333,
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
                    'metric_target_drift': [
                        4862.9411666666665,
                        4790.5815,
                        4793.349333333334,
                        4838.256166666667,
                        4799.1335,
                        4852.636666666666,
                        4875.456666666667,
                        4867.589,
                        4885.108,
                        4787.094166666667,
                    ],
                    'statistical_target_drift': [
                        0.014249999999999985,
                        0.016566666666666674,
                        0.010066666666666668,
                        0.011916666666666659,
                        0.008666666666666656,
                        0.1716833333333333,
                        0.18011666666666665,
                        0.17906666666666665,
                        0.18323333333333333,
                        0.1873833333333333,
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
    calc = TargetDistributionCalculator(
        y_true='y_true',
        problem_type=ProblemType.REGRESSION,
        **calculator_opts,
    ).fit(reference)
    results = calc.calculate(analysis.join(analysis_targets))
    sut = results.data[results.data['period'] == 'analysis'].reset_index(drop=True)

    pd.testing.assert_frame_equal(expected, sut[['key', 'metric_target_drift', 'statistical_target_drift']])


@pytest.mark.parametrize(
    'calculator_opts, expected',
    [
        (
            {'chunk_size': 10000},
            pd.DataFrame(
                {
                    'key': ['[0:9999]', '[10000:19999]', '[20000:29999]', '[30000:39999]', '[40000:49999]'],
                    'metric_target_drift': [0.5044, 0.4911, 0.501, 0.5028, 0.5041],
                    'statistical_target_drift': [
                        0.7552537772547374,
                        2.3632451058508988,
                        0.061653341622277646,
                        0.3328533514553376,
                        0.6630536280238508,
                    ],
                }
            ),
        ),
        (
            {'chunk_size': 10000, 'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': ['[0:9999]', '[10000:19999]', '[20000:29999]', '[30000:39999]', '[40000:49999]'],
                    'metric_target_drift': [0.5044, 0.4911, 0.501, 0.5028, 0.5041],
                    'statistical_target_drift': [
                        0.7552537772547374,
                        2.3632451058508988,
                        0.061653341622277646,
                        0.3328533514553376,
                        0.6630536280238508,
                    ],
                }
            ),
        ),
        (
            {'chunk_number': 5},
            pd.DataFrame(
                {
                    'key': ['[0:9999]', '[10000:19999]', '[20000:29999]', '[30000:39999]', '[40000:49999]'],
                    'metric_target_drift': [0.5044, 0.4911, 0.501, 0.5028, 0.5041],
                    'statistical_target_drift': [
                        0.7552537772547374,
                        2.3632451058508988,
                        0.061653341622277646,
                        0.3328533514553376,
                        0.6630536280238508,
                    ],
                }
            ),
        ),
        (
            {'chunk_number': 5, 'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': ['[0:9999]', '[10000:19999]', '[20000:29999]', '[30000:39999]', '[40000:49999]'],
                    'metric_target_drift': [0.5044, 0.4911, 0.501, 0.5028, 0.5041],
                    'statistical_target_drift': [
                        0.7552537772547374,
                        2.3632451058508988,
                        0.061653341622277646,
                        0.3328533514553376,
                        0.6630536280238508,
                    ],
                }
            ),
        ),
        (
            {'chunk_period': 'Y', 'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {
                    'key': ['2017', '2018', '2019', '2020', '2021'],
                    'metric_target_drift': [
                        0.5175686591276252,
                        0.49144221838927954,
                        0.5021347565043363,
                        0.5030052090289836,
                        0.4,
                    ],
                    'statistical_target_drift': [
                        5.760397194889025,
                        3.0356541961122385,
                        0.2909492514342073,
                        0.5271435212375012,
                        0.09826781851967825,
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
                    'metric_target_drift': [
                        0.5172,
                        0.4916,
                        0.4858,
                        0.4964,
                        0.5026,
                        0.4994,
                        0.505,
                        0.5006,
                        0.4964,
                        0.5118,
                    ],
                    'statistical_target_drift': [
                        5.574578416652949,
                        1.1261313647806923,
                        3.3976543904089302,
                        0.17136216283219585,
                        0.15396546757525365,
                        8.909097694730939e-05,
                        0.5126563744826438,
                        0.015056370086959939,
                        0.17136216283219585,
                        2.666406909476474,
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
                    'metric_target_drift': [
                        0.5172,
                        0.4916,
                        0.4858,
                        0.4964,
                        0.5026,
                        0.4994,
                        0.505,
                        0.5006,
                        0.4964,
                        0.5118,
                    ],
                    'statistical_target_drift': [
                        5.574578416652949,
                        1.1261313647806923,
                        3.3976543904089302,
                        0.17136216283219585,
                        0.15396546757525365,
                        8.909097694730939e-05,
                        0.5126563744826438,
                        0.015056370086959939,
                        0.17136216283219585,
                        2.666406909476474,
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
    calc = TargetDistributionCalculator(
        y_true='work_home_actual',
        problem_type=ProblemType.CLASSIFICATION_BINARY,
        **calculator_opts,
    ).fit(reference)
    results = calc.calculate(analysis.merge(analysis_targets, on='identifier'))
    sut = results.data[results.data['period'] == 'analysis'].reset_index(drop=True)

    pd.testing.assert_frame_equal(expected, sut[['key', 'metric_target_drift', 'statistical_target_drift']])


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
                        0.12834899947890172,
                        3.1960676394816177,
                        1.295948474797905,
                        19.16124656547084,
                        18.025854609422936,
                        24.018246053152254,
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
                        0.12834899947890172,
                        3.1960676394816177,
                        1.295948474797905,
                        19.16124656547084,
                        18.025854609422936,
                        24.018246053152254,
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
                        0.26657487437501814,
                        2.312782400721795,
                        12.420850432581522,
                        21.51752691617733,
                        24.77164649048273,
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
                        0.26657487437501814,
                        2.312782400721795,
                        12.420850432581522,
                        21.51752691617733,
                        24.77164649048273,
                    ],
                }
            ),
        ),
        (
            {'chunk_period': 'Y', 'timestamp_column_name': 'timestamp'},
            pd.DataFrame(
                {'key': ['2020', '2021'], 'statistical_target_drift': [19.18793594439608, 0.4207915478721409]}
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
                        0.5215450181058865,
                        2.1122555296182584,
                        0.9401078333614571,
                        2.130103897306355,
                        2.2209947008941855,
                        14.42105157991354,
                        6.009302835706899,
                        19.749168564900494,
                        14.08710527642606,
                        12.884655612509915,
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
                        0.5215450181058865,
                        2.1122555296182584,
                        0.9401078333614571,
                        2.130103897306355,
                        2.2209947008941855,
                        14.42105157991354,
                        6.009302835706899,
                        19.749168564900494,
                        14.08710527642606,
                        12.884655612509915,
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
    calc = TargetDistributionCalculator(
        y_true='y_true',
        problem_type=ProblemType.CLASSIFICATION_MULTICLASS,
        **calculator_opts,
    ).fit(reference)
    results = calc.calculate(analysis.merge(analysis_targets, on='identifier')).filter(period='analysis')
    sut = results.data[results.data['period'] == 'analysis'].reset_index(drop=True)

    pd.testing.assert_frame_equal(expected, sut[['key', 'statistical_target_drift']])


@pytest.mark.parametrize(
    'calc_args, plot_args',
    [
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'target_drift', 'plot_reference': False}),
        ({}, {'kind': 'target_drift', 'plot_reference': False}),
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'target_drift', 'plot_reference': True}),
        ({}, {'kind': 'target_drift', 'plot_reference': True}),
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'target_distribution', 'plot_reference': False}),
        ({}, {'kind': 'target_distribution', 'plot_reference': False}),
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'target_distribution', 'plot_reference': True}),
        ({}, {'kind': 'target_distribution', 'plot_reference': True}),
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
def test_multiclass_classification_result_plots_raise_no_exceptions(calc_args, plot_args):  # noqa: D103
    reference, analysis, analysis_targets = load_synthetic_multiclass_classification_dataset()
    calc = TargetDistributionCalculator(
        y_true='y_true', problem_type=ProblemType.CLASSIFICATION_MULTICLASS, **calc_args
    ).fit(reference)
    sut = calc.calculate(analysis.merge(analysis_targets, on='identifier'))

    try:
        _ = sut.plot(**plot_args)
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


@pytest.mark.parametrize(
    'calc_args, plot_args',
    [
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'target_drift', 'plot_reference': False}),
        ({}, {'kind': 'target_drift', 'plot_reference': False}),
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'target_drift', 'plot_reference': True}),
        ({}, {'kind': 'target_drift', 'plot_reference': True}),
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'target_distribution', 'plot_reference': False}),
        ({}, {'kind': 'target_distribution', 'plot_reference': False}),
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'target_distribution', 'plot_reference': True}),
        ({}, {'kind': 'target_distribution', 'plot_reference': True}),
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
def test_binary_classification_result_plots_raise_no_exceptions(calc_args, plot_args):  # noqa: D103
    reference, analysis, analysis_targets = load_synthetic_binary_classification_dataset()
    calc = TargetDistributionCalculator(
        y_true='work_home_actual', problem_type=ProblemType.CLASSIFICATION_BINARY, **calc_args
    ).fit(reference)
    sut = calc.calculate(analysis.merge(analysis_targets, on='identifier'))

    try:
        _ = sut.plot(**plot_args)
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


@pytest.mark.parametrize(
    'calc_args, plot_args',
    [
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'target_drift', 'plot_reference': False}),
        ({}, {'kind': 'target_drift', 'plot_reference': False}),
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'target_drift', 'plot_reference': True}),
        ({}, {'kind': 'target_drift', 'plot_reference': True}),
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'target_distribution', 'plot_reference': False}),
        ({}, {'kind': 'target_distribution', 'plot_reference': False}),
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'target_distribution', 'plot_reference': True}),
        ({}, {'kind': 'target_distribution', 'plot_reference': True}),
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
def test_regression_result_plots_raise_no_exceptions(calc_args, plot_args):  # noqa: D103
    reference, analysis, analysis_targets = load_synthetic_car_price_dataset()
    calc = TargetDistributionCalculator(y_true='y_true', problem_type=ProblemType.REGRESSION, **calc_args).fit(
        reference
    )
    sut = calc.calculate(analysis.join(analysis_targets))

    try:
        _ = sut.plot(**plot_args)
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")
