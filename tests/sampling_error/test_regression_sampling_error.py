#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

import numpy as np
import pandas as pd

import nannyml.sampling_error.regression as rse


def test_mae_sampling_error():
    np.random.seed(1)
    sample_size = 50
    chunk = np.random.random(sample_size)
    population_size = 10000
    y_true = np.random.random(population_size)
    y_pred = np.random.random(population_size)

    components = rse.mae_sampling_error_components(pd.Series(y_true), pd.Series(y_pred))
    sampling_error = rse.mae_sampling_error(components, chunk)
    assert np.round(sampling_error, 4) == 0.0334


def test_mape_sampling_error():
    np.random.seed(1)
    sample_size = 50
    chunk = np.random.random(sample_size)
    population_size = 10000
    y_true = np.random.random(population_size)
    y_pred = np.random.random(population_size)

    components = rse.mape_sampling_error_components(pd.Series(y_true), pd.Series(y_pred))
    sampling_error = rse.mape_sampling_error(components, chunk)
    assert np.round(sampling_error, 4) == 14.0368


def test_mse_sampling_error():
    np.random.seed(1)
    sample_size = 50
    chunk = np.random.random(sample_size)
    population_size = 10000
    y_true = np.random.random(population_size)
    y_pred = np.random.random(population_size)

    components = rse.mse_sampling_error_components(pd.Series(y_true), pd.Series(y_pred))
    sampling_error = rse.mse_sampling_error(components, chunk)
    assert np.round(sampling_error, 4) == 0.028


def test_msle_sampling_error():
    np.random.seed(1)
    sample_size = 50
    chunk = np.random.random(sample_size)
    population_size = 10000
    y_true = np.random.random(population_size)
    y_pred = np.random.random(population_size)

    components = rse.msle_sampling_error_components(pd.Series(y_true), pd.Series(y_pred))
    sampling_error = rse.msle_sampling_error(components, chunk)
    assert np.round(sampling_error, 4) == 0.0134


def test_rmse_sampling_error():
    np.random.seed(1)
    sample_size = 50
    chunk = np.random.random(sample_size)
    population_size = 10000
    y_true = np.random.random(population_size)
    y_pred = np.random.random(population_size)

    components = rse.rmse_sampling_error_components(pd.Series(y_true), pd.Series(y_pred))
    sampling_error = rse.rmse_sampling_error(components, chunk)
    assert np.round(sampling_error, 4) == 0.0346


def test_rmsle_sampling_error():
    np.random.seed(1)
    sample_size = 50
    chunk = np.random.random(sample_size)
    population_size = 10000
    y_true = np.random.random(population_size)
    y_pred = np.random.random(population_size)

    components = rse.rmsle_sampling_error_components(pd.Series(y_true), pd.Series(y_pred))
    sampling_error = rse.rmsle_sampling_error(components, chunk)
    assert np.round(sampling_error, 4) == 0.0241
