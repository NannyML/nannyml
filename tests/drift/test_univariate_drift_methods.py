"""Unit tests for the UnivariateDriftCalculator methods."""
import numpy as np
import pandas as pd

from nannyml.drift.univariate.methods import JensenShannonDistance


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
