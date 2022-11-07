"""Unit tests for the UnivariateDriftCalculator methods."""
import numpy as np
import pandas as pd
import pytest
from nannyml.drift.univariate.methods import JensenShannonDistance, InfinityNormDistance
import nannyml as nml


def test_js_for_0_distance():
    np.random.seed(1)
    reference = pd.Series(np.random.choice(np.linspace(0, 2, 6), 10_000))
    js = JensenShannonDistance()
    js.fit(reference)
    distance = js.calculate(reference)
    assert distance == 0


def test_js_for_both_continuous():
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 10_000))
    analysis = pd.Series(np.random.normal(0, 1, 1000))
    js = JensenShannonDistance()
    js.fit(reference)
    distance = js.calculate(analysis)
    assert np.round(distance, 2) == 0.05


def test_js_for_quasi_continuous():
    np.random.seed(1)
    reference = pd.Series(np.random.choice(np.linspace(0, 2, 6), 10_000))
    analysis = pd.Series(np.random.choice(np.linspace(0, 2, 3), 1000))
    js = JensenShannonDistance()
    js.fit(reference)
    distance = js.calculate(analysis)
    assert np.round(distance, 2) == 0.73


def test_js_for_categorical():
    np.random.seed(1)
    reference = pd.Series(np.random.choice(['a', 'b', 'c', 'd'], 10_000))
    analysis = pd.Series(np.random.choice(['a', 'b', 'c', 'e'], 1000))
    js = JensenShannonDistance()
    js.fit(reference)
    distance = js.calculate(analysis)
    assert np.round(distance, 2) == 0.5


def test_infinity_norm_for_new_category():
    reference = pd.Series(['a', 'a', 'b', 'b', 'c', 'c'])
    analysis = pd.Series(['a', 'a', 'b', 'b', 'c', 'c', 'd'])
    infnorm = InfinityNormDistance()
    infnorm.fit(reference)
    distance = infnorm.calculate(analysis)
    assert np.round(distance, 2) == 0.14


def test_infinity_norm_for_no_change():
    reference = pd.Series(['a', 'a', 'b', 'b', 'c', 'c'])
    analysis = pd.Series(['a', 'a', 'b', 'b', 'c', 'c'])
    infnorm = InfinityNormDistance()
    infnorm.fit(reference)
    distance = infnorm.calculate(analysis)
    assert np.round(distance, 2) == 0.0


def test_infinity_norm_for_total_change():
    reference = pd.Series(['a', 'a', 'b', 'b', 'c', 'c'])
    analysis = pd.Series(['b', 'b', 'b', 'b', 'b'])
    infnorm = InfinityNormDistance()
    infnorm.fit(reference)
    distance = infnorm.calculate(analysis)
    assert np.round(distance, 2) == 0.67


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


@pytest.mark.parametrize(
    'continuous_methods, categorical_methods',
    [
        (
            [],
            ['chi2'],
        ),
        (
            ['jensen_shannon'],
            ['jensen_shannon'],
        ),
        (
            [],
            ['infinity_norm'],
        ),
        (
            ['kolmogorov_smirnov'],
            [],
        ),
    ],
    ids=[
        'feature_drift_with_ks_and_chi2',
        'feature_drift_with_js_and_js',
        'feature_drift_with_none_and_infinitynorm',
        'feature_drift_with_ks_and_none',
    ],
)
def test_result_plots_raise_no_exceptions(sample_drift_data, continuous_methods, categorical_methods):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    ana_data = sample_drift_data.loc[sample_drift_data['period'] == 'analysis']
    try:
        calc = nml.UnivariateDriftCalculator(
            column_names=['f1', 'f3'],
            timestamp_column_name='timestamp',
            continuous_methods=continuous_methods,
            categorical_methods=categorical_methods,
        ).fit(ref_data)
        calc.calculate(ana_data)
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")
