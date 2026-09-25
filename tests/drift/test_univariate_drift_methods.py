"""Unit tests for the UnivariateDriftCalculator methods."""
import numpy as np
import pandas as pd
import pytest

from nannyml.chunk import CountBasedChunker, DefaultChunker
from nannyml.drift.univariate.methods import (
    CategoricalHellingerDistance,
    CategoricalJensenShannonDistance,
    Chi2Statistic,
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


# ************* Chi2 Tests *************


@pytest.mark.parametrize(
    'analysis, expected_alert',
    [
        (pd.Series(['a'] * 20 + ['b'] * 80, name='A'), True),
        (pd.Series(['a'] * 78 + ['b'] * 22, name='A'), False),
        (pd.Series(['a'] * 80 + ['b'] * 20, name='A'), False),
    ],
    ids=['drift', 'no_drift', 'same_distribution_as_reference'],
)
def test_chi2_alert_is_based_on_p_value(analysis, expected_alert):  # noqa: D103
    reference = pd.Series(['a'] * 800 + ['b'] * 200, name='A')
    chi2 = Chi2Statistic(chunker=chunker, threshold=threshold).fit(reference)
    value = chi2.calculate(analysis)
    assert chi2.alert(value) == expected_alert


@pytest.mark.parametrize(
    'previous_analysis',
    [
        pd.Series(['a'] * 20 + ['b'] * 80, name='A'),
        pd.Series(['a'] * 78 + ['b'] * 22, name='A'),
    ],
    ids=['after_drift', 'after_no_drift'],
)
def test_chi2_does_not_alert_for_chunk_with_only_missing_values(previous_analysis):  # noqa: D103
    reference = pd.Series(['a'] * 800 + ['b'] * 200, name='A')
    chi2 = Chi2Statistic(chunker=chunker, threshold=threshold).fit(reference)
    chi2.alert(chi2.calculate(previous_analysis))
    value = chi2.calculate(pd.Series([None] * 100, name='A'))
    assert np.isnan(value)
    assert not chi2.alert(value)


def test_chi2_does_not_alert_for_first_chunk_with_only_missing_values_after_fit():  # noqa: D103
    # the last reference chunk only contains 'b' values, so it has drifted from the full reference data
    reference = pd.Series(['a'] * 500 + ['b'] * 500, name='A')
    chi2 = Chi2Statistic(chunker=CountBasedChunker(2), threshold=threshold).fit(reference)
    value = chi2.calculate(pd.Series([None] * 100, name='A'))
    assert np.isnan(value)
    assert not chi2.alert(value)


def test_chi2_alerts_for_chunk_with_some_missing_values_after_chunk_with_only_missing_values():  # noqa: D103
    reference = pd.Series(['a'] * 800 + ['b'] * 200, name='A')
    chi2 = Chi2Statistic(chunker=chunker, threshold=threshold).fit(reference)
    chi2.alert(chi2.calculate(pd.Series([None] * 100, name='A')))
    value = chi2.calculate(pd.Series(['a'] * 20 + ['b'] * 80 + [None] * 100, name='A'))
    assert chi2.alert(value)


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
