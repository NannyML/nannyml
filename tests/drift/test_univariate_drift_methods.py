"""Unit tests for the UnivariateDriftCalculator methods."""
import numpy as np
import pandas as pd
import pytest

from nannyml import UnivariateDriftCalculator
from nannyml.chunk import CountBasedChunker, DefaultChunker
from nannyml.drift.univariate.methods import (
    CategoricalHellingerDistance,
    CategoricalJensenShannonDistance,
    ContinuousHellingerDistance,
    ContinuousJensenShannonDistance,
    KolmogorovSmirnovStatistic,
    LInfinityDistance,
    WassersteinDistance,
)
from nannyml.thresholds import ConstantThreshold

# ************* JS Tests *************

chunker = CountBasedChunker(1)
threshold = ConstantThreshold(lower=None, upper=0.1)


def test_js_for_0_distance():  # noqa: D103
    np.random.seed(1)
    reference = pd.Series(np.random.choice(np.linspace(0, 2, 6), 10_000), name='A')
    js = ContinuousJensenShannonDistance(chunker=chunker, threshold=threshold)
    js.fit(reference)
    distance = js.calculate(reference)
    assert distance == 0


def test_js_for_both_continuous():  # noqa: D103
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 10_000), name='A')
    analysis = pd.Series(np.random.normal(0, 1, 1000), name='A')
    js = ContinuousJensenShannonDistance(chunker=chunker, threshold=threshold)
    js.fit(reference)
    distance = js.calculate(analysis)
    assert np.round(distance, 2) == 0.05


def test_js_for_quasi_continuous():  # noqa: D103
    np.random.seed(1)
    reference = pd.Series(np.random.choice(np.linspace(0, 2, 6), 10_000), name='A')
    analysis = pd.Series(np.random.choice(np.linspace(0, 2, 3), 1000), name='A')
    js = ContinuousJensenShannonDistance(chunker=chunker, threshold=threshold)
    js.fit(reference)
    distance = js.calculate(analysis)
    assert np.round(distance, 2) == 0.73


def test_js_for_categorical():  # noqa: D103
    np.random.seed(1)
    reference = pd.Series(np.random.choice(['a', 'b', 'c', 'd'], 10_000), name='A')
    analysis = pd.Series(np.random.choice(['a', 'b', 'c', 'e'], 1000), name='A')
    js = CategoricalJensenShannonDistance(chunker=chunker, threshold=threshold)
    js.fit(reference)
    distance = js.calculate(analysis)
    assert np.round(distance, 2) == 0.5


def test_l_infinity_for_new_category():  # noqa: D103
    reference = pd.Series(['a', 'a', 'b', 'b', 'c', 'c'], name='A')
    analysis = pd.Series(['a', 'a', 'b', 'b', 'c', 'c', 'd'], name='A')
    infnorm = LInfinityDistance(chunker=chunker, threshold=threshold)
    infnorm.fit(reference)
    distance = infnorm.calculate(analysis)
    assert np.round(distance, 2) == 0.14


def test_l_infinity_for_no_change():  # noqa: D103
    reference = pd.Series(['a', 'a', 'b', 'b', 'c', 'c'], name='A')
    analysis = pd.Series(['a', 'a', 'b', 'b', 'c', 'c'], name='A')
    infnorm = LInfinityDistance(chunker=chunker, threshold=threshold)
    infnorm.fit(reference)
    distance = infnorm.calculate(analysis)
    assert np.round(distance, 2) == 0.0


def test_l_infinity_for_total_change():  # noqa: D103
    reference = pd.Series(['a', 'a', 'b', 'b', 'c', 'c'], name='A')
    analysis = pd.Series(['b', 'b', 'b', 'b', 'b'], name='A')
    infnorm = LInfinityDistance(chunker=chunker, threshold=threshold)
    infnorm.fit(reference)
    distance = infnorm.calculate(analysis)
    assert np.round(distance, 2) == 0.67


# ************* Wasserstein Tests *************


def test_wasserstein_both_continuous_0_distance():  # noqa: D103
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 10_000), name='A')
    analysis = reference
    wass_dist = WassersteinDistance(chunker=chunker, threshold=threshold)
    wass_dist = wass_dist.fit(reference).calculate(analysis)
    wass_dist = np.round(wass_dist, 2)
    assert wass_dist == 0


def test_wasserstein_both_continuous_positive_means_small_drift():  # noqa: D103
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 10000), name='A')
    analysis = pd.Series(np.random.normal(1, 1, 1000), name='A')
    wass_dist = WassersteinDistance(chunker=chunker, threshold=threshold)
    wass_dist = wass_dist.fit(reference).calculate(analysis)
    wass_dist = np.round(wass_dist, 2)
    assert wass_dist == 1.01


def test_wasserstein_both_continuous_analysis_with_neg_mean_medium_drift():  # noqa: D103
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 100000), name='A')
    analysis = pd.Series(np.random.normal(-4, 1, 1000), name='A')
    wass_dist = WassersteinDistance(chunker=chunker, threshold=threshold)
    wass_dist = wass_dist.fit(reference).calculate(analysis)
    wass_dist = np.round(wass_dist, 2)
    assert wass_dist == 3.99


def test_wasserstein_both_continuous_analysis_estimate_with_out_of_reference_drift():  # noqa: D103
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 15_000), name='A')
    analysis = pd.Series(np.random.normal(0, 10, 1_000_000), name='A')
    wass_dist = WassersteinDistance(chunker=chunker, threshold=threshold)
    wass_dist = wass_dist.fit(reference).calculate(analysis)
    wass_dist = np.round(wass_dist, 3)
    assert wass_dist == 7.180


# ************* Hellinger Tests *************


def test_hellinger_complete_overlap():  # noqa: D103
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 10_000), name='A')
    analysis = reference
    hell_dist = ContinuousHellingerDistance(chunker=chunker, threshold=threshold).fit(reference).calculate(analysis)
    hell_dist = np.round(hell_dist, 2)
    assert hell_dist == 0


def test_hellinger_no_overlap():  # noqa: D103
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 10_000), name='A')
    analysis = pd.Series(np.random.normal(7, 1, 10_000), name='A')
    hell_dist = ContinuousHellingerDistance(chunker=chunker, threshold=threshold).fit(reference).calculate(analysis)
    hell_dist = np.round(hell_dist, 2)
    assert hell_dist == 1


def test_hellinger_both_continuous_analysis_with_small_drift():  # noqa: D103
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 10_000), name='A')
    analysis = pd.Series(np.random.normal(-2, 1, 10_000), name='A')
    hell_dist = ContinuousHellingerDistance(chunker=chunker, threshold=threshold).fit(reference).calculate(analysis)
    hell_dist = np.round(hell_dist, 2)
    assert hell_dist == 0.63


def test_hellinger_for_quasi_continuous():  # noqa: D103
    np.random.seed(1)
    reference = pd.Series(np.random.choice(np.linspace(0, 2, 6), 10_000), name='A')
    analysis = pd.Series(np.random.choice(np.linspace(0, 2, 3), 1000), name='A')
    hell_dist = ContinuousHellingerDistance(chunker=chunker, threshold=threshold)
    hell_dist.fit(reference)
    distance = hell_dist.calculate(analysis)
    assert np.round(distance, 2) == 0.72


def test_hellinger_for_categorical():  # noqa: D103
    np.random.seed(1)
    reference = pd.Series(np.random.choice(['a', 'b', 'c', 'd'], 10_000), name='A')
    analysis = pd.Series(np.random.choice(['a', 'b', 'c', 'e'], 1000), name='A')
    hell_dist = CategoricalHellingerDistance(chunker=chunker, threshold=threshold)
    hell_dist.fit(reference)
    distance = hell_dist.calculate(analysis)
    assert np.round(distance, 2) == 0.5


@pytest.mark.parametrize(
    'method',
    [
        KolmogorovSmirnovStatistic(chunker=DefaultChunker(), threshold=ConstantThreshold(lower=-1, upper=None)),
        LInfinityDistance(chunker=DefaultChunker(), threshold=ConstantThreshold(lower=-1, upper=None)),
        ContinuousJensenShannonDistance(chunker=DefaultChunker(), threshold=ConstantThreshold(lower=-1, upper=None)),
        WassersteinDistance(chunker=DefaultChunker(), threshold=ConstantThreshold(lower=-1, upper=None)),
        ContinuousHellingerDistance(chunker=DefaultChunker(), threshold=ConstantThreshold(lower=-1, upper=None)),
    ],
)
def test_method_logs_warning_when_lower_threshold_is_overridden_by_metric_limits(caplog, method):  # noqa: D103
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 1000), name='A')
    method.fit(reference)

    assert (
        f'{method.display_name} lower threshold value -1 overridden by '
        f'lower threshold value limit {method.lower_threshold_value_limit}' in caplog.messages
    )


@pytest.mark.parametrize(
    'method',
    [
        KolmogorovSmirnovStatistic(chunker=DefaultChunker(), threshold=ConstantThreshold(upper=2)),
    ],
)
def test_method_logs_warning_when_upper_threshold_is_overridden_by_metric_limits(caplog, method):  # noqa: D103
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 1000), name='A')
    method.fit(reference)

    assert (
        f'{method.display_name} upper threshold value 2 overridden by '
        f'upper threshold value limit {method.upper_threshold_value_limit}' in caplog.messages
    )


@pytest.mark.parametrize('method_class', [KolmogorovSmirnovStatistic, WassersteinDistance])
@pytest.mark.parametrize('calculation_method,reference_size', [('estimated', 100), ('exact', 10_000)])
def test_method_uses_nested_computation_params(method_class, calculation_method, reference_size):
    method = method_class(chunker=chunker, threshold=threshold)
    params = {method.column_name: {'calculation_method': calculation_method, 'n_bins': 8}}
    method = method_class(chunker=chunker, threshold=threshold, computation_params=params)
    reference = pd.Series(np.linspace(0, 1, reference_size), name='A')

    method.fit(reference)

    assert method.calculation_method == calculation_method
    assert method.n_bins == 8
    if calculation_method == 'exact':
        pd.testing.assert_series_equal(method._reference_data, reference)
    else:
        assert method._reference_data is None
        assert len(method._ref_rel_freqs) == 8
    assert np.isfinite(method.calculate(reference + 0.2))
    assert params == {method.column_name: {'calculation_method': calculation_method, 'n_bins': 8}}


@pytest.mark.parametrize('method_class', [KolmogorovSmirnovStatistic, WassersteinDistance])
@pytest.mark.parametrize('params', [{}, {'unrelated': {'calculation_method': 'exact', 'n_bins': 8}}])
def test_method_preserves_default_computation_params(method_class, params):
    method = method_class(chunker=chunker, threshold=threshold, computation_params=params)

    assert method.calculation_method == 'auto'
    assert method.n_bins == 10_000


@pytest.mark.parametrize('method_class', [KolmogorovSmirnovStatistic, WassersteinDistance])
@pytest.mark.parametrize('settings', [{}, {'n_bins': 8}, {'calculation_method': 'exact'}])
def test_method_fills_missing_computation_params(method_class, settings):
    method = method_class(chunker=chunker, threshold=threshold)
    params = {method.column_name: settings}
    method = method_class(chunker=chunker, threshold=threshold, computation_params=params)

    assert method.calculation_method == settings.get('calculation_method', 'auto')
    assert method.n_bins == settings.get('n_bins', 10_000)


def test_calculator_keeps_computation_params_separate_for_each_method():
    reference = pd.DataFrame({'A': np.linspace(0, 1, 100)})
    calculator = UnivariateDriftCalculator(
        column_names=['A'],
        continuous_methods=['kolmogorov_smirnov', 'wasserstein'],
        chunk_size=10,
        computation_params={
            'kolmogorov_smirnov': {'calculation_method': 'estimated', 'n_bins': 8},
            'wasserstein': {'calculation_method': 'exact', 'n_bins': 16},
        },
    ).fit(reference)
    ks, wasserstein = calculator._column_to_models_mapping['A']

    assert ks.calculation_method == 'estimated'
    assert len(ks._ref_rel_freqs) == 8
    assert wasserstein.calculation_method == 'exact'
    assert wasserstein.n_bins == 16
    pd.testing.assert_series_equal(wasserstein._reference_data, reference['A'])
    result = calculator.calculate(reference + 0.2)
    assert np.isfinite(result.filter(period='analysis').data['A']['kolmogorov_smirnov']['value']).all()
    assert np.isfinite(result.filter(period='analysis').data['A']['wasserstein']['value']).all()
