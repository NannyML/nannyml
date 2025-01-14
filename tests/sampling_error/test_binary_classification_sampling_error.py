#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Tests."""

import numpy as np
import pandas as pd

import nannyml.sampling_error.binary_classification as bse


def test_auroc_sampling_error():  # noqa: D103
    np.random.seed(1)
    sample_size = 50
    chunk = np.random.random(sample_size)
    population_size = 10000
    y_true = np.random.binomial(1, 0.5, population_size)
    y_pred_proba = np.linspace(0, 1, population_size)

    components = bse.auroc_sampling_error_components(pd.Series(y_true), pd.Series(y_pred_proba))
    sampling_error = bse.auroc_sampling_error(components, chunk)
    assert np.round(sampling_error, 4) == 0.0575


def test_auroc_sampling_error_nan():  # noqa: D103
    np.random.seed(1)
    sample_size = 50
    chunk = np.random.random(sample_size)

    components = np.nan, np.nan
    sampling_error = bse.auroc_sampling_error(components, chunk)
    assert np.isnan(sampling_error)


def test_f1_sampling_error():  # noqa: D103
    np.random.seed(1)
    sample_size = 50
    chunk = np.random.random(sample_size)
    population_size = 10000
    y_true = np.random.binomial(1, 0.5, population_size)
    y_pred = np.random.binomial(1, 0.5, population_size)

    components = bse.f1_sampling_error_components(pd.Series(y_true), pd.Series(y_pred))
    sampling_error = bse.f1_sampling_error(components, chunk)
    assert np.round(sampling_error, 4) == 0.1153


def test_precision_sampling_error():  # noqa: D103
    np.random.seed(1)
    sample_size = 50
    chunk = np.random.random(sample_size)
    population_size = 10000
    y_true = np.random.binomial(1, 0.5, population_size)
    y_pred = np.random.binomial(1, 0.5, population_size)

    components = bse.precision_sampling_error_components(pd.Series(y_true), pd.Series(y_pred))
    sampling_error = bse.precision_sampling_error(components, chunk)
    assert np.round(sampling_error, 4) == 0.0989


def test_recall_sampling_error():  # noqa: D103
    np.random.seed(1)
    sample_size = 50
    chunk = np.random.random(sample_size)
    population_size = 10000
    y_true = np.random.binomial(1, 0.5, population_size)
    y_pred = np.random.binomial(1, 0.5, population_size)

    components = bse.recall_sampling_error_components(pd.Series(y_true), pd.Series(y_pred))
    sampling_error = bse.recall_sampling_error(components, chunk)
    assert np.round(sampling_error, 4) == 0.0998


def test_specificity_sampling_error():  # noqa: D103
    np.random.seed(1)
    sample_size = 50
    chunk = np.random.random(sample_size)
    population_size = 10000
    y_true = np.random.binomial(1, 0.5, population_size)
    y_pred = np.random.binomial(1, 0.5, population_size)

    components = bse.specificity_sampling_error_components(pd.Series(y_true), pd.Series(y_pred))
    sampling_error = bse.specificity_sampling_error(components, chunk)
    assert np.round(sampling_error, 4) == 0.1001


def test_accuracy_sampling_error():  # noqa: D103
    np.random.seed(1)
    sample_size = 50
    chunk = np.random.random(sample_size)
    population_size = 10000
    y_true = np.random.binomial(1, 0.5, population_size)
    y_pred = np.random.binomial(1, 0.5, population_size)

    components = bse.accuracy_sampling_error_components(pd.Series(y_true), pd.Series(y_pred))
    sampling_error = bse.accuracy_sampling_error(components, chunk)
    assert np.round(sampling_error, 4) == 0.0707


def test_ap_sampling_error_when_nan():  # noqa: D103
    comp1 = np.nan
    comp2 = 0
    data = pd.DataFrame({'y_true': [0, 1, 1], 'y_pred_proba': [0.4, 0.6, 0.7]})

    sampling_error = bse.ap_sampling_error((comp1, comp2), data)
    assert np.isnan(sampling_error)
