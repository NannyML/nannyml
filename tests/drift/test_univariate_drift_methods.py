"""Unit tests for the UnivariateDriftCalculator methods."""
import numpy as np
import pandas as pd
import pytest

from nannyml.chunk import CountBasedChunker, DefaultChunker
from nannyml.drift.univariate.methods import (
    HellingerDistance,
    JensenShannonDistance,
    KolmogorovSmirnovStatistic,
    LInfinityDistance,
    WassersteinDistance,
)
from nannyml.thresholds import ConstantThreshold

# ************* JS Tests *************

chunker = CountBasedChunker(1)
threshold = ConstantThreshold(lower=None, upper=0.1)


def test_js_for_0_distance():
    np.random.seed(1)
    reference = pd.Series(np.random.choice(np.linspace(0, 2, 6), 10_000), name='A')
    js = JensenShannonDistance(chunker=chunker, threshold=threshold)
    js.fit(reference)
    distance = js.calculate(reference)
    assert distance == 0


def test_js_for_both_continuous():
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 10_000), name='A')
    analysis = pd.Series(np.random.normal(0, 1, 1000), name='A')
    js = JensenShannonDistance(chunker=chunker, threshold=threshold)
    js.fit(reference)
    distance = js.calculate(analysis)
    assert np.round(distance, 2) == 0.05


def test_js_for_quasi_continuous():
    np.random.seed(1)
    reference = pd.Series(np.random.choice(np.linspace(0, 2, 6), 10_000), name='A')
    analysis = pd.Series(np.random.choice(np.linspace(0, 2, 3), 1000), name='A')
    js = JensenShannonDistance(chunker=chunker, threshold=threshold)
    js.fit(reference)
    distance = js.calculate(analysis)
    assert np.round(distance, 2) == 0.73


def test_js_for_categorical():
    np.random.seed(1)
    reference = pd.Series(np.random.choice(['a', 'b', 'c', 'd'], 10_000), name='A')
    analysis = pd.Series(np.random.choice(['a', 'b', 'c', 'e'], 1000), name='A')
    js = JensenShannonDistance(chunker=chunker, threshold=threshold)
    js.fit(reference)
    distance = js.calculate(analysis)
    assert np.round(distance, 2) == 0.5


def test_l_infinity_for_new_category():
    reference = pd.Series(['a', 'a', 'b', 'b', 'c', 'c'], name='A')
    analysis = pd.Series(['a', 'a', 'b', 'b', 'c', 'c', 'd'], name='A')
    infnorm = LInfinityDistance(chunker=chunker, threshold=threshold)
    infnorm.fit(reference)
    distance = infnorm.calculate(analysis)
    assert np.round(distance, 2) == 0.14


def test_l_infinity_for_no_change():
    reference = pd.Series(['a', 'a', 'b', 'b', 'c', 'c'], name='A')
    analysis = pd.Series(['a', 'a', 'b', 'b', 'c', 'c'], name='A')
    infnorm = LInfinityDistance(chunker=chunker, threshold=threshold)
    infnorm.fit(reference)
    distance = infnorm.calculate(analysis)
    assert np.round(distance, 2) == 0.0


def test_l_infinity_for_total_change():
    reference = pd.Series(['a', 'a', 'b', 'b', 'c', 'c'], name='A')
    analysis = pd.Series(['b', 'b', 'b', 'b', 'b'], name='A')
    infnorm = LInfinityDistance(chunker=chunker, threshold=threshold)
    infnorm.fit(reference)
    distance = infnorm.calculate(analysis)
    assert np.round(distance, 2) == 0.67


# ************* Wasserstein Tests *************


def test_wasserstein_both_continuous_0_distance():
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 10_000), name='A')
    analysis = reference
    wass_dist = WassersteinDistance(chunker=chunker, threshold=threshold)
    wass_dist = wass_dist.fit(reference).calculate(analysis)
    wass_dist = np.round(wass_dist, 2)
    assert wass_dist == 0


def test_wasserstein_both_continuous_positive_means_small_drift():
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 10000), name='A')
    analysis = pd.Series(np.random.normal(1, 1, 1000), name='A')
    wass_dist = WassersteinDistance(chunker=chunker, threshold=threshold)
    wass_dist = wass_dist.fit(reference).calculate(analysis)
    wass_dist = np.round(wass_dist, 2)
    assert wass_dist == 1.01


def test_wasserstein_both_continuous_analysis_with_neg_mean_medium_drift():
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 100000), name='A')
    analysis = pd.Series(np.random.normal(-4, 1, 1000), name='A')
    wass_dist = WassersteinDistance(chunker=chunker, threshold=threshold)
    wass_dist = wass_dist.fit(reference).calculate(analysis)
    wass_dist = np.round(wass_dist, 2)
    assert wass_dist == 3.99


# ************* Hellinger Tests *************


def test_hellinger_complete_overlap():
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 10_000), name='A')
    analysis = reference
    hell_dist = HellingerDistance(chunker=chunker, threshold=threshold).fit(reference).calculate(analysis)
    hell_dist = np.round(hell_dist, 2)
    assert hell_dist == 0


def test_hellinger_no_overlap():
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 10_000), name='A')
    analysis = pd.Series(np.random.normal(7, 1, 10_000), name='A')
    hell_dist = HellingerDistance(chunker=chunker, threshold=threshold).fit(reference).calculate(analysis)
    hell_dist = np.round(hell_dist, 2)
    assert hell_dist == 1


def test_hellinger_both_continuous_analysis_with_small_drift():
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 10_000), name='A')
    analysis = pd.Series(np.random.normal(-2, 1, 10_000), name='A')
    hell_dist = HellingerDistance(chunker=chunker, threshold=threshold).fit(reference).calculate(analysis)
    hell_dist = np.round(hell_dist, 2)
    assert hell_dist == 0.63


def test_hellinger_for_quasi_continuous():
    np.random.seed(1)
    reference = pd.Series(np.random.choice(np.linspace(0, 2, 6), 10_000), name='A')
    analysis = pd.Series(np.random.choice(np.linspace(0, 2, 3), 1000), name='A')
    hell_dist = HellingerDistance(chunker=chunker, threshold=threshold)
    hell_dist.fit(reference)
    distance = hell_dist.calculate(analysis)
    assert np.round(distance, 2) == 0.72


def test_hellinger_for_categorical():
    np.random.seed(1)
    reference = pd.Series(np.random.choice(['a', 'b', 'c', 'd'], 10_000), name='A')
    analysis = pd.Series(np.random.choice(['a', 'b', 'c', 'e'], 1000), name='A')
    hell_dist = HellingerDistance(chunker=chunker, threshold=threshold)
    hell_dist.fit(reference)
    distance = hell_dist.calculate(analysis)
    assert np.round(distance, 2) == 0.5


@pytest.mark.parametrize(
    'method',
    [
        KolmogorovSmirnovStatistic(chunker=DefaultChunker(), threshold=ConstantThreshold(lower=-1, upper=None)),
        LInfinityDistance(chunker=DefaultChunker(), threshold=ConstantThreshold(lower=-1, upper=None)),
        JensenShannonDistance(chunker=DefaultChunker(), threshold=ConstantThreshold(lower=-1, upper=None)),
        WassersteinDistance(chunker=DefaultChunker(), threshold=ConstantThreshold(lower=-1, upper=None)),
        HellingerDistance(chunker=DefaultChunker(), threshold=ConstantThreshold(lower=-1, upper=None)),
    ],
)
def test_method_logs_warning_when_lower_threshold_is_overridden_by_metric_limits(caplog, method):
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
def test_method_logs_warning_when_upper_threshold_is_overridden_by_metric_limits(caplog, method):
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 1000), name='A')
    method.fit(reference)

    assert (
        f'{method.display_name} upper threshold value 2 overridden by '
        f'upper threshold value limit {method.upper_threshold_value_limit}' in caplog.messages
    )
