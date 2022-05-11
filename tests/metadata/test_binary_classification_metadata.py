#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Tests concerning :class:`nannyml.metadata.binary_classification.BinaryClassificationMetadata`."""

from typing import Tuple

import pandas as pd
import pytest

from nannyml.datasets import load_synthetic_binary_classification_dataset
from nannyml.metadata import BinaryClassificationMetadata, FeatureType, ModelMetadata, ModelType, extract_metadata
from nannyml.metadata.base import (
    NML_METADATA_PARTITION_COLUMN_NAME,
    NML_METADATA_TARGET_COLUMN_NAME,
    NML_METADATA_TIMESTAMP_COLUMN_NAME,
)
from nannyml.metadata.binary_classification import (
    NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME,
    NML_METADATA_PREDICTION_COLUMN_NAME,
    _guess_predicted_probabilities,
    _guess_predictions,
)


@pytest.fixture
def data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, tgt_df = load_synthetic_binary_classification_dataset()
    ref_df['y_pred'] = ref_df['y_pred_proba'].map(lambda p: p >= 0.8).astype(int)
    ana_df['y_pred'] = ana_df['y_pred_proba'].map(lambda p: p >= 0.8).astype(int)

    return ref_df, ana_df, tgt_df


@pytest.fixture
def metadata(data) -> ModelMetadata:  # noqa: D103
    md = extract_metadata(data[0], model_type='classification_binary')
    md.target_column_name = 'work_home_actual'
    return md


def test_model_metadata_creation_with_defaults_has_correct_properties():  # noqa: D103
    sut = BinaryClassificationMetadata()

    assert sut.name is None
    assert sut.model_type == ModelType.CLASSIFICATION_BINARY
    assert sut.features is not None
    assert len(sut.features) == 0
    assert sut.prediction_column_name is None
    assert sut.predicted_probability_column_name is None
    assert sut.target_column_name == 'target'
    assert sut.partition_column_name == 'partition'
    assert sut.timestamp_column_name == 'date'


def test_model_metadata_creation_sets_correct_values_for_special_attributes():  # noqa: D103
    sut = BinaryClassificationMetadata(prediction_column_name='pred', predicted_probability_column_name='pred_proba')
    assert sut.prediction_column_name == 'pred'
    assert sut.predicted_probability_column_name == 'pred_proba'


def test_to_dict_contains_all_properties(metadata):  # noqa: D103
    sut = metadata.to_dict()
    assert sut['prediction_column_name'] == 'y_pred'
    assert sut['predicted_probability_column_name'] == 'y_pred_proba'


def test_to_pd_contains_all_properties(metadata):  # noqa: D103
    sut = metadata.to_df()
    assert sut.loc[sut['label'] == 'prediction_column_name', 'column_name'].iloc[0] == 'y_pred'
    assert sut.loc[sut['label'] == 'predicted_probability_column_name', 'column_name'].iloc[0] == 'y_pred_proba'


@pytest.mark.parametrize(
    'col,expected',
    [
        ('p', True),
        ('y_pred', True),
        ('pred', True),
        ('prediction', True),
        ('out', True),
        ('output', True),
        ('nope', False),
    ],
)
def test_guess_predictions_yields_correct_results(col, expected):  # noqa: D103
    sut = _guess_predictions(data=pd.DataFrame(columns=[col]))
    assert col == sut[0] if expected else len(sut) == 0


@pytest.mark.parametrize(
    'col,expected',
    [('y_pred_proba', True), ('nope', False)],
)
def test_guess_predicted_probabilities_yields_correct_results(col, expected):  # noqa: D103
    sut = _guess_predicted_probabilities(data=pd.DataFrame(columns=[col]))
    assert col == sut[0] if expected else len(sut) == 0


def test_setting_prediction_column_name_after_extracting_metadata_updates_the_features_list(  # noqa: D103
    data: pd.DataFrame,
):
    data[0].rename(columns={'y_pred': 'foo'}, inplace=True)
    md = extract_metadata(data[0], model_type='classification_binary')
    md.prediction_column_name = 'foo'
    assert md.prediction_column_name == 'foo'
    assert md.feature(column='foo') is None


def test_setting_predicted_probability_column_name_after_extracting_metadata_updates_the_features_list(  # noqa: D103
    data: pd.DataFrame,
):
    data[0].rename(columns={'y_pred_proba': 'score'}, inplace=True)
    md = extract_metadata(data[0], model_type='classification_binary')
    md.prediction_column_name = 'score'
    assert md.prediction_column_name == 'score'
    assert md.feature(column='score') is None


def test_enrich_copies_each_metadata_column_to_new_fixed_column():  # noqa: D103
    data = pd.DataFrame(columns=['identity', 'prediction', 'actual', 'partition', 'ts', 'feat1', 'feat2'])
    md = extract_metadata(data, model_name='model', model_type='classification_binary')
    sut = md.enrich(data).columns

    assert NML_METADATA_TIMESTAMP_COLUMN_NAME in sut
    assert NML_METADATA_PREDICTION_COLUMN_NAME in sut
    assert NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME in sut
    assert NML_METADATA_TARGET_COLUMN_NAME in sut
    assert NML_METADATA_PARTITION_COLUMN_NAME in sut


def test_enrich_works_on_copy_of_data_by_default():  # noqa: D103
    data = pd.DataFrame(columns=['identity', 'prediction', 'actual', 'partition', 'ts', 'feat1', 'feat2'])
    old_column_count = len(data.columns)
    md = extract_metadata(data, model_name='model', model_type='classification_binary')
    sut = md.enrich(data).columns

    assert len(sut) == len(data.columns) + 5
    assert len(data.columns) == old_column_count

    assert NML_METADATA_TIMESTAMP_COLUMN_NAME in sut
    assert NML_METADATA_PREDICTION_COLUMN_NAME in sut
    assert NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME in sut
    assert NML_METADATA_TARGET_COLUMN_NAME in sut
    assert NML_METADATA_PARTITION_COLUMN_NAME in sut
    assert 'prediction' in sut
    assert 'actual' in sut
    assert 'partition' in sut
    assert 'feat1' in sut
    assert 'feat2' in sut


def test_enrich_adds_nan_prediction_column_if_no_prediction_column_in_original_data(data):  # noqa: D103
    md = extract_metadata(data[0], model_type='classification_binary')
    analysis_data = data[1].drop(columns=[md.prediction_column_name])
    sut = md.enrich(analysis_data)
    assert NML_METADATA_PREDICTION_COLUMN_NAME in sut.columns
    assert sut[NML_METADATA_PREDICTION_COLUMN_NAME].isna().all()


def test_enrich_adds_nan_predicted_probability_column_if_no_predicted_probability_in_original_data(  # noqa: D103
    data,
):
    md = extract_metadata(data[0], model_type='classification_binary')
    analysis_data = data[1].drop(columns=[md.prediction_column_name])
    sut = md.enrich(analysis_data)
    assert NML_METADATA_PREDICTION_COLUMN_NAME in sut.columns
    assert sut[NML_METADATA_PREDICTION_COLUMN_NAME].isna().all()


def test_enrich_adds_nan_ground_truth_column_if_no_ground_truth_in_original_data(data, metadata):  # noqa: D103
    sut = metadata.enrich(data[1])
    assert NML_METADATA_TARGET_COLUMN_NAME in sut.columns
    assert sut[NML_METADATA_TARGET_COLUMN_NAME].isna().all()


def test_complete_returns_all_missing_properties_when_metadata_is_incomplete(data, metadata):  # noqa: D103
    metadata.timestamp_column_name = None
    metadata.prediction_column_name = None
    metadata.predicted_probability_column_name = None
    sut = metadata.is_complete()
    assert sut[0] is False
    assert 'timestamp_column_name' in sut[1]
    assert 'prediction_column_name' in sut[1]
    assert 'predicted_probability_column_name' in sut[1]


def test_complete_returns_complete_if_prediction_is_none_but_predicted_probabilities_are_not(  # noqa: D103
    data, metadata
):
    metadata.prediction_column_name = None
    sut = metadata.is_complete()
    assert sut[0] is True
    assert 'prediction_column_name' not in sut[1]
    assert 'predicted_probability_column_name' not in sut[1]


def test_complete_returns_complete_if_predicted_probabilities_is_none_but_prediction_is_not(  # noqa: D103
    data, metadata
):
    metadata.predicted_probability_column_name = None
    sut = metadata.is_complete()
    assert sut[0] is True
    assert 'prediction_column_name' not in sut[1]
    assert 'predicted_probability_column_name' not in sut[1]


def test_complete_returns_incomplete_and_both_prediction_props_when_none_of_both_set(data, metadata):  # noqa: D103
    metadata.prediction_column_name = None
    metadata.predicted_probability_column_name = None
    sut = metadata.is_complete()
    assert sut[0] is False
    assert 'prediction_column_name' in sut[1]
    assert 'predicted_probability_column_name' in sut[1]


def test_complete_returns_false_when_features_of_unknown_feature_type_exist(data, metadata):  # noqa: D103
    # Rig the data
    metadata.features[0].feature_type = FeatureType.UNKNOWN
    metadata.features[1].feature_type = FeatureType.UNKNOWN

    is_complete, missing = metadata.is_complete()
    assert not is_complete
    assert metadata.features[0].label in missing
    assert metadata.features[1].label in missing


def test_extract_metadata_should_set_binary_classification_properties(data):  # noqa: D103
    sut = extract_metadata(data[0], model_type='classification_binary')
    assert sut is not None

    # check binary classification properties
    assert sut.prediction_column_name == 'y_pred'
    assert sut.predicted_probability_column_name == 'y_pred_proba'

    # check base properties
    assert sut.target_column_name is None
    assert sut.partition_column_name == 'partition'
    assert sut.timestamp_column_name == 'timestamp'
