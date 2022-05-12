#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  #
#  License: Apache Software License 2.0

"""Tests concerning :class:`nannyml.metadata.multiclass_classification.MulticlassClassificationMetadata`."""

from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from nannyml.datasets import load_synthetic_binary_classification_dataset
from nannyml.metadata import ModelType, MulticlassClassificationMetadata, extract_metadata
from nannyml.metadata.base import (
    NML_METADATA_PARTITION_COLUMN_NAME,
    NML_METADATA_TARGET_COLUMN_NAME,
    NML_METADATA_TIMESTAMP_COLUMN_NAME,
)
from nannyml.metadata.multiclass_classification import (
    NML_METADATA_PREDICTION_COLUMN_NAME,
    _extract_class_to_column_mapping,
    _guess_predicted_probabilities,
    _guess_predictions,
)


@pytest.fixture
def data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, tgt_df = load_synthetic_binary_classification_dataset()
    ref_df.drop(columns=['y_pred_proba'], inplace=True)
    ana_df.drop(columns=['y_pred_proba'], inplace=True)

    # setup result classes
    classes = ['remote', 'onsite', 'hybrid']
    for clazz in classes:
        ref_df[f'y_pred_proba_{clazz}'] = np.random.randn(len(ref_df))
        ana_df[f'y_pred_proba_{clazz}'] = np.random.randn(len(ana_df))

    ref_df['y_pred'] = np.random.choice(classes, size=len(ref_df))
    ana_df['y_pred'] = np.random.choice(classes, size=len(ana_df))

    return ref_df, ana_df, tgt_df


@pytest.fixture
def metadata(data) -> MulticlassClassificationMetadata:  # noqa: D103
    md = extract_metadata(data[0], model_type='classification_multiclass', exclude_columns=['identifier'])
    md.target_column_name = 'work_home_actual'
    return md


def test_model_metadata_creation_with_defaults_has_correct_properties():  # noqa: D103
    sut = MulticlassClassificationMetadata()

    assert sut.name is None
    assert sut.model_type == ModelType.CLASSIFICATION_MULTICLASS
    assert sut.prediction_column_name is None
    assert sut.predicted_probabilities_column_names is None


def test_model_metadata_creation_sets_correct_values_for_special_attributes():  # noqa: D103
    sut = MulticlassClassificationMetadata(
        prediction_column_name='pred', predicted_probabilities_column_names={'A': 'pred_proba_A', 'B': 'pred_proba_B'}
    )
    assert sut.prediction_column_name == 'pred'
    assert sut.predicted_probabilities_column_names['A'] == 'pred_proba_A'
    assert sut.predicted_probabilities_column_names['B'] == 'pred_proba_B'


def test_to_dict_contains_all_properties(metadata):  # noqa: D103
    sut = metadata.to_dict()
    assert sut['prediction_column_name'] == 'y_pred'
    assert sut['predicted_probabilities_column_names'] == {
        'remote': 'y_pred_proba_remote',
        'onsite': 'y_pred_proba_onsite',
        'hybrid': 'y_pred_proba_hybrid',
    }


def test_to_pd_contains_all_properties(metadata):  # noqa: D103
    sut = metadata.to_df()
    assert sut.loc[sut['label'] == 'prediction_column_name', 'column_name'].iloc[0] == 'y_pred'
    assert (
        sut.loc[sut['label'] == 'predicted_probability_column_name_remote', 'column_name'].iloc[0]
        == 'y_pred_proba_remote'
    )
    assert (
        sut.loc[sut['label'] == 'predicted_probability_column_name_hybrid', 'column_name'].iloc[0]
        == 'y_pred_proba_hybrid'
    )
    assert (
        sut.loc[sut['label'] == 'predicted_probability_column_name_onsite', 'column_name'].iloc[0]
        == 'y_pred_proba_onsite'
    )


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
    [('y_pred_proba', False), ('nope', False), ('y_pred_proba_0', True), ('y_pred_proba_X', True)],
)
def test_guess_predicted_probabilities_yields_correct_results(col, expected):  # noqa: D103
    sut = _guess_predicted_probabilities(data=pd.DataFrame(columns=[col]))
    assert col == sut[0] if expected else len(sut) == 0


@pytest.mark.parametrize(
    'clazz,column_name',
    [(0, 'y_pred_proba_0'), ('A', 'y_pred_proba_A'), ('Foo', 'y_pred_proba_Foo'), ('Foo_Bar', 'y_pred_proba_Foo_Bar')],
)
def test_extract_class_to_column_mapping_finds_present_class_names(clazz, column_name):  # noqa: D103
    sut = _extract_class_to_column_mapping([column_name])
    assert clazz in sut
    assert sut[clazz] == column_name


@pytest.mark.parametrize('clazz,column_name', [(0, 'y_pred_proba'), ('A', 'y_pred_probability_A')])
def test_extract_class_to_column_mapping_skips_non_present_class_names(clazz, column_name):  # noqa: D103
    sut = _extract_class_to_column_mapping([column_name])
    assert clazz not in sut


def test_extract_metadata_should_set_multiclass_classification_properties(data):  # noqa: D103
    sut = extract_metadata(data[0], model_type='classification_multiclass')
    assert sut is not None

    # check binary classification properties
    assert sut.prediction_column_name == 'y_pred'
    assert sut.predicted_probabilities_column_names['remote'] == 'y_pred_proba_remote'
    assert sut.predicted_probabilities_column_names['onsite'] == 'y_pred_proba_onsite'
    assert sut.predicted_probabilities_column_names['hybrid'] == 'y_pred_proba_hybrid'

    # check base properties
    assert sut.target_column_name is None
    assert sut.partition_column_name == 'partition'
    assert sut.timestamp_column_name == 'timestamp'


def test_complete_returns_complete_if_prediction_is_none_but_predicted_probabilities_are_not(  # noqa: D103
    data, metadata
):
    metadata.prediction_column_name = None
    sut = metadata.is_complete()
    assert sut[0] is True
    assert 'prediction_column_name' not in sut[1]
    assert 'predicted_probabilities_column_names' not in sut[1]


def test_complete_returns_complete_if_predicted_probabilities_is_none_but_prediction_is_not(metadata):  # noqa: D103
    metadata.predicted_probabilities_column_names = None
    sut = metadata.is_complete()
    assert sut[0] is True
    assert 'prediction_column_name' not in sut[1]
    assert 'predicted_probabilities_column_names' not in sut[1]


def test_is_complete_raises_missing_metadata_exception_if_both_predicted_class_and_probabilities_missing(  # noqa: D103
    metadata,
):
    metadata.prediction_column_name = None
    metadata.predicted_probabilities_column_names = None
    ok, missing = metadata.is_complete()
    assert not ok
    assert 'predicted_probabilities_column_names' in missing
    assert 'prediction_column_name' in missing


def test_setting_prediction_column_name_after_extracting_metadata_updates_the_features_list(  # noqa: D103
    data: pd.DataFrame,
):
    data[0].rename(columns={'y_pred': 'foo'}, inplace=True)
    md = extract_metadata(data[0], model_type='classification_multiclass', exclude_columns=['identifier'])
    md.prediction_column_name = 'foo'
    assert md.prediction_column_name == 'foo'
    assert md.feature(column='foo') is None


def test_setting_predicted_probabilities_column_names_after_extracting_metadata_updates_the_features_list(  # noqa: D103
    data: pd.DataFrame,
):
    data[0].rename(
        columns={
            'y_pred_proba_remote': 'score_remote',
            'y_pred_proba_onsite': 'score_onsite',
            'y_pred_proba_hybrid': 'score_hybrid',
        },
        inplace=True,
    )
    md = extract_metadata(data[0], model_type='classification_multiclass', exclude_columns=['identifier'])
    md.predicted_probabilities_column_names = {
        'remote': 'score_remote',
        'onsite': 'score_onsite',
        'hybrid': 'score_hybrid',
    }
    assert md.predicted_probabilities_column_names['remote'] == 'score_remote'
    assert md.predicted_probabilities_column_names['onsite'] == 'score_onsite'
    assert md.predicted_probabilities_column_names['hybrid'] == 'score_hybrid'
    assert md.feature(column='score_remote') is None
    assert md.feature(column='score_onsite') is None
    assert md.feature(column='score_hybrid') is None


def test_enrich_copies_each_metadata_column_to_new_fixed_column(data, metadata):  # noqa: D103
    md = extract_metadata(data[0], model_type='classification_multiclass', exclude_columns=['identifier'])
    sut = md.enrich(data[0]).columns

    assert NML_METADATA_PREDICTION_COLUMN_NAME in sut
    assert 'nml_meta_predicted_proba_remote' in sut
    assert 'nml_meta_predicted_proba_onsite' in sut
    assert 'nml_meta_predicted_proba_hybrid' in sut

    assert NML_METADATA_TIMESTAMP_COLUMN_NAME in sut
    assert NML_METADATA_TARGET_COLUMN_NAME in sut
    assert NML_METADATA_PARTITION_COLUMN_NAME in sut


def test_enrich_adds_nan_prediction_column_if_no_prediction_column_in_original_data(data, metadata):  # noqa: D103
    analysis_data = data[1].drop(columns=[metadata.prediction_column_name])
    sut = metadata.enrich(analysis_data)
    assert NML_METADATA_PREDICTION_COLUMN_NAME in sut.columns
    assert sut[NML_METADATA_PREDICTION_COLUMN_NAME].isna().all()


def test_enrich_adds_nan_predicted_probabilities_columns_if_no_predicted_probability_in_original_data(  # noqa: D103
    data, metadata
):
    analysis_data = data[1].drop(columns=list(metadata.predicted_probabilities_column_names.values()))
    sut = metadata.enrich(analysis_data)
    assert 'nml_meta_predicted_proba_remote' in sut.columns
    assert sut['nml_meta_predicted_proba_remote'].isna().all()

    assert 'nml_meta_predicted_proba_onsite' in sut.columns
    assert sut['nml_meta_predicted_proba_onsite'].isna().all()

    assert 'nml_meta_predicted_proba_hybrid' in sut.columns
    assert sut['nml_meta_predicted_proba_hybrid'].isna().all()


def test_metadata_columns_returns_correct_metadata_columns(metadata):  # noqa: D103
    sut = metadata.metadata_columns
    assert len(sut) == 3 + 1 + 3
    assert 'nml_meta_prediction' in sut
    assert 'nml_meta_predicted_proba_remote' in sut
    assert 'nml_meta_predicted_proba_onsite' in sut
    assert 'nml_meta_predicted_proba_hybrid' in sut


@pytest.mark.parametrize(
    'mapping,ok,missing',
    [
        ({'remote': 'y_pred_proba_remote', 'onsite': 'y_pred_proba_onsite', 'hybrid': 'y_pred_proba_hybrid'}, True, []),
        ({'remote': 'y_pred_proba_remote', 'onsite': 'y_pred_proba_onsite'}, False, ['hybrid']),
        ({'onsite': 'y_pred_proba_onsite'}, False, ['hybrid', 'remote']),
    ],
)
def test_validate_predicted_class_labels_in_class_probability_mapping(  # noqa: D103
    metadata: MulticlassClassificationMetadata, data, mapping, ok, missing
):
    metadata.predicted_probabilities_column_names = mapping
    _ok, _missing = metadata.validate_predicted_class_labels_in_class_probability_mapping(data[0])
    assert _ok == ok
    assert _missing == missing
