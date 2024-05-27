#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Tests."""

import numpy as np
from sklearn.preprocessing import LabelBinarizer

import nannyml.sampling_error.multiclass_classification as mse


def test_multiclass_auroc_sampling_error():  # noqa: D103
    np.random.seed(1)
    n_classes = 3
    chunk = [i for i in range(50)]
    population_size = 10_000

    y_true = np.random.randint(0, n_classes, population_size)
    lb = LabelBinarizer()
    y_true = lb.fit_transform(y_true)

    y_pred_proba = abs(y_true - 0.49) + np.random.normal(0, 0.01, np.shape(y_true))
    y_pred_proba = (y_pred_proba.T / (y_pred_proba.sum(axis=1))).T

    list_of_y_pred_probas = list(y_pred_proba.T)
    list_of_y_trues = list(y_true.T)

    components = mse.auroc_sampling_error_components(list_of_y_trues, list_of_y_pred_probas)
    sampling_error = mse.auroc_sampling_error(components, chunk)
    assert np.round(sampling_error, 4) == 0.0149


def test_multiclass_auroc_sampling_error_with_mean_of_y_true_over_point_5():  # noqa: D103
    np.random.seed(1)
    n_classes = 3
    chunk = [i for i in range(50)]
    population_size = 10_000

    y_true = np.random.randint(0, n_classes, population_size)
    y_true[0 : population_size * 2 // 3] = 0

    lb = LabelBinarizer()
    y_true = lb.fit_transform(y_true)

    y_pred_proba = abs(y_true - 0.49) + np.random.normal(0, 0.01, np.shape(y_true))
    y_pred_proba = (y_pred_proba.T / (y_pred_proba.sum(axis=1))).T

    list_of_y_pred_probas = list(y_pred_proba.T)
    list_of_y_trues = list(y_true.T)

    components = mse.auroc_sampling_error_components(list_of_y_trues, list_of_y_pred_probas)
    sampling_error = mse.auroc_sampling_error(components, chunk)
    assert np.round(sampling_error, 4) == 0.0237


def test_multiclass_f1_sampling_error():  # noqa: D103
    np.random.seed(1)
    n_classes = 3
    chunk = [i for i in range(50)]
    population_size = 10_000

    y_true = np.random.randint(0, n_classes, population_size)
    y_pred = np.random.randint(0, n_classes, population_size)

    lb = LabelBinarizer()
    y_true = lb.fit_transform(y_true)
    y_pred = lb.transform(y_pred)

    list_of_y_preds = list(y_pred.T)
    list_of_y_trues = list(y_true.T)

    components = mse.f1_sampling_error_components(list_of_y_trues, list_of_y_preds)
    sampling_error = mse.f1_sampling_error(components, chunk)
    assert np.round(sampling_error, 4) == 0.0440


def test_multiclass_precision_sampling_error():  # noqa: D103
    np.random.seed(1)
    n_classes = 3
    chunk = [i for i in range(50)]
    population_size = 10_000

    y_true = np.random.randint(0, n_classes, population_size)
    y_pred = np.random.randint(0, n_classes, population_size)

    lb = LabelBinarizer()
    y_true = lb.fit_transform(y_true)
    y_pred = lb.transform(y_pred)

    list_of_y_preds = list(y_pred.T)
    list_of_y_trues = list(y_true.T)

    components = mse.precision_sampling_error_components(list_of_y_trues, list_of_y_preds)
    sampling_error = mse.precision_sampling_error(components, chunk)
    assert np.round(sampling_error, 4) == 0.0668


def test_multiclass_recall_sampling_error():  # noqa: D103
    np.random.seed(1)
    n_classes = 3
    chunk = [i for i in range(50)]
    population_size = 10_000

    y_true = np.random.randint(0, n_classes, population_size)
    y_pred = np.random.randint(0, n_classes, population_size)

    lb = LabelBinarizer()
    y_true = lb.fit_transform(y_true)
    y_pred = lb.transform(y_pred)

    list_of_y_preds = list(y_pred.T)
    list_of_y_trues = list(y_true.T)

    components = mse.recall_sampling_error_components(list_of_y_trues, list_of_y_preds)
    sampling_error = mse.recall_sampling_error(components, chunk)
    assert np.round(sampling_error, 4) == 0.0668


def test_multiclass_specificity_sampling_error():  # noqa: D103
    np.random.seed(1)
    n_classes = 3
    chunk = [i for i in range(50)]
    population_size = 10_000

    y_true = np.random.randint(0, n_classes, population_size)
    y_pred = np.random.randint(0, n_classes, population_size)

    lb = LabelBinarizer()
    y_true = lb.fit_transform(y_true)
    y_pred = lb.transform(y_pred)

    list_of_y_preds = list(y_pred.T)
    list_of_y_trues = list(y_true.T)

    components = mse.specificity_sampling_error_components(list_of_y_trues, list_of_y_preds)
    sampling_error = mse.specificity_sampling_error(components, chunk)
    assert np.round(sampling_error, 4) == 0.0471


def test_multiclass_accuracy_sampling_error():  # noqa: D103
    np.random.seed(1)
    n_classes = 3
    chunk = [i for i in range(50)]
    population_size = 10_000

    y_true = np.random.randint(0, n_classes, population_size)
    y_pred = np.random.randint(0, n_classes, population_size)

    lb = LabelBinarizer()
    y_true = lb.fit_transform(y_true)
    y_pred = lb.transform(y_pred)

    components = mse.accuracy_sampling_error_components(y_true, y_pred)
    sampling_error = mse.accuracy_sampling_error(components, chunk)
    assert np.round(sampling_error, 4) == 0.0668
