#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit testing for CBPE."""
from typing import Tuple

import pandas as pd
import pytest

from nannyml import InvalidArgumentsException, MissingMetadataException
from nannyml.datasets import load_synthetic_sample
from nannyml.metadata import ModelMetadata, extract_metadata
from nannyml.performance_estimation import CBPE


@pytest.fixture
def data() -> Tuple[pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, _ = load_synthetic_sample()
    ref_df['y_pred'] = ref_df['y_pred_proba'].apply(lambda p: p >= 0.8)
    return ref_df, ana_df


@pytest.fixture
def metadata(data) -> ModelMetadata:  # noqa: D103
    md = extract_metadata(data[0], exclude_columns=['identifier'])
    md.target_column_name = 'work_home_actual'
    return md


def test_cbpe_will_calibrate_scores_when_needed(metadata, data):  # noqa: D103
    ref_df = data[0]

    sut = CBPE(model_metadata=metadata, metrics=['roc_auc'])
    sut.fit(ref_df)

    assert sut.needs_calibration is True


def test_cbpe_will_not_calibrate_scores_when_not_needed(metadata, data):  # noqa: D103
    ref_df = data[0]
    # If predictions equal targets no calibration will be required
    ref_df[metadata.predicted_probability_column_name] = ref_df[metadata.target_column_name]

    sut = CBPE(model_metadata=metadata, metrics=['roc_auc'])
    sut.fit(ref_df)

    assert sut.needs_calibration is False


def test_cbpe_will_not_fail_on_work_from_home_sample(metadata, data):  # noqa: D103
    reference, analysis = data
    try:
        estimator = CBPE(model_metadata=metadata, metrics=['roc_auc'])
        estimator.fit(reference)
        _ = estimator.estimate(analysis)
    except Exception as exc:
        pytest.fail(f'unexpected exception was raised: {exc}')


def test_cbpe_raises_invalid_arguments_exception_when_giving_invalid_metric_value(metadata):  # noqa: D103
    with pytest.raises(InvalidArgumentsException, match="unknown 'metric' value: 'foo'."):
        _ = CBPE(model_metadata=metadata, metrics=['f1', 'foo'])


def test_cbpe_raises_invalid_arguments_exception_when_given_empty_metrics_list(metadata):  # noqa: D103
    with pytest.raises(
        InvalidArgumentsException, match="no metrics provided. Please provide a non-empty list of metrics."
    ):
        _ = CBPE(model_metadata=metadata, metrics=[])


def test_cbpe_raises_invalid_arguments_exception_when_given_none_metrics_list(metadata):  # noqa: D103
    with pytest.raises(
        InvalidArgumentsException, match="no metrics provided. Please provide a non-empty list of metrics."
    ):
        _ = CBPE(model_metadata=metadata, metrics=None)


def test_cbpe_raises_missing_metadata_exception_when_predictions_are_required_but_not_given(  # noqa: D103
    metadata, data
):
    metadata.prediction_column_name = None

    reference, _ = data
    estimator = CBPE(model_metadata=metadata, metrics=['f1'])  # requires predictions!
    with pytest.raises(
        MissingMetadataException,
        match="missing value for 'prediction_column_name'. "
        "Please ensure predicted label values are specified and present "
        "in the sample.",
    ):
        estimator.fit(reference)


def test_cbpe_raises_missing_metadata_exception_when_predicted_probabilities_are_required_but_not_given(  # noqa: D103
    metadata, data
):
    reference, _ = data

    metadata.predicted_probability_column_name = None

    estimator = CBPE(model_metadata=metadata, metrics=['roc_auc'])
    with pytest.raises(
        MissingMetadataException,
        match="missing value for 'predicted_probability_column_name'. "
        "Please ensure predicted label values are specified and present "
        "in the sample.",
    ):
        estimator.fit(reference)


@pytest.mark.parametrize('metric', ['roc_auc', 'f1', 'precision', 'recall', 'specificity', 'accuracy'])
def test_cbpe_runs_for_all_metrics(data, metric):  # noqa: D103
    reference, analysis = data
    reference['y_pred'] = reference['y_pred_proba'].apply(lambda p: int(p >= 0.8))
    metadata = extract_metadata(reference, exclude_columns=['identifier'])
    metadata.target_column_name = 'work_home_actual'
    try:
        estimator = CBPE(model_metadata=metadata, chunk_size=5000, metrics=[metric]).fit(reference)
        _ = estimator.estimate(pd.concat([reference, analysis]))
    except Exception as e:
        pytest.fail(f'an unexpected exception occurred: {e}')
