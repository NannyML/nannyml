"""Unit tests for the UnivariateDriftCalculator methods."""
import numpy as np
import pandas as pd

from nannyml.drift.univariate.methods import (
    HellingerDistance,
    JensenShannonDistance,
    LInfinityDistance,
    WassersteinDistance,
)

# ************* JS Tests *************


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


def test_l_infinity_for_new_category():
    reference = pd.Series(['a', 'a', 'b', 'b', 'c', 'c'])
    analysis = pd.Series(['a', 'a', 'b', 'b', 'c', 'c', 'd'])
    infnorm = LInfinityDistance()
    infnorm.fit(reference)
    distance = infnorm.calculate(analysis)
    assert np.round(distance, 2) == 0.14


def test_l_infinity_for_no_change():
    reference = pd.Series(['a', 'a', 'b', 'b', 'c', 'c'])
    analysis = pd.Series(['a', 'a', 'b', 'b', 'c', 'c'])
    infnorm = LInfinityDistance()
    infnorm.fit(reference)
    distance = infnorm.calculate(analysis)
    assert np.round(distance, 2) == 0.0


def test_l_infinity_for_total_change():
    reference = pd.Series(['a', 'a', 'b', 'b', 'c', 'c'])
    analysis = pd.Series(['b', 'b', 'b', 'b', 'b'])
    infnorm = LInfinityDistance()
    infnorm.fit(reference)
    distance = infnorm.calculate(analysis)
    assert np.round(distance, 2) == 0.67


# ************* Wasserstein Tests *************


def test_wasserstein_both_continuous_0_distance():
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 10_000))
    analysis = reference
    wass_dist = WassersteinDistance()
    wass_dist = wass_dist.fit(reference).calculate(analysis)
    wass_dist = np.round(wass_dist, 2)
    assert wass_dist == 0


def test_wasserstein_both_continuous_positive_means_small_drift():
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 10000))
    analysis = pd.Series(np.random.normal(1, 1, 1000))
    wass_dist = WassersteinDistance()
    wass_dist = wass_dist.fit(reference).calculate(analysis)
    wass_dist = np.round(wass_dist, 2)
    assert wass_dist == 1.01


def test_wasserstein_both_continuous_analysis_with_neg_mean_medium_drift():
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 100000))
    analysis = pd.Series(np.random.normal(-4, 1, 1000))
    wass_dist = WassersteinDistance()
    wass_dist = wass_dist.fit(reference).calculate(analysis)
    wass_dist = np.round(wass_dist, 2)
    assert wass_dist == 3.99


# ************* Hellinger Tests *************


def test_hellinger_complete_overlap():
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 10_000))
    analysis = reference
    hell_dist = HellingerDistance().fit(reference).calculate(analysis)
    hell_dist = np.round(hell_dist, 2)
    assert hell_dist == 0


def test_hellinger_no_overlap():
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 10_000))
    analysis = pd.Series(np.random.normal(7, 1, 10_000))
    hell_dist = HellingerDistance().fit(reference).calculate(analysis)
    hell_dist = np.round(hell_dist, 2)
    assert hell_dist == 1


def test_hellinger_both_continuous_analysis_with_small_drift():
    np.random.seed(1)
    reference = pd.Series(np.random.normal(0, 1, 10_000))
    analysis = pd.Series(np.random.normal(-2, 1, 10_000))
    hell_dist = HellingerDistance().fit(reference).calculate(analysis)
    hell_dist = np.round(hell_dist, 2)
    assert hell_dist == 0.63


def test_hellinger_for_quasi_continuous():
    np.random.seed(1)
    reference = pd.Series(np.random.choice(np.linspace(0, 2, 6), 10_000))
    analysis = pd.Series(np.random.choice(np.linspace(0, 2, 3), 1000))
    hell_dist = HellingerDistance()
    hell_dist.fit(reference)
    distance = hell_dist.calculate(analysis)
    assert np.round(distance, 2) == 0.72


def test_hellinger_for_categorical():
    np.random.seed(1)
    reference = pd.Series(np.random.choice(['a', 'b', 'c', 'd'], 10_000))
    analysis = pd.Series(np.random.choice(['a', 'b', 'c', 'e'], 1000))
    hell_dist = HellingerDistance()
    hell_dist.fit(reference)
    distance = hell_dist.calculate(analysis)
    assert np.round(distance, 2) == 0.5
