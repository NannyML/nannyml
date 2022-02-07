#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit tests for metadata module."""

import math
import re

import numpy as np
import pandas as pd
import pytest

from nannyml.metadata import (
    NML_METADATA_GROUND_TRUTH_COLUMN_NAME,
    NML_METADATA_IDENTIFIER_COLUMN_NAME,
    NML_METADATA_PARTITION_COLUMN_NAME,
    NML_METADATA_PREDICTION_COLUMN_NAME,
    NML_METADATA_TIMESTAMP_COLUMN_NAME,
    Feature,
    FeatureType,
    ModelMetadata,
    _guess_features,
    _guess_ground_truths,
    _guess_identifiers,
    _guess_partitions,
    _guess_predictions,
    _guess_timestamps,
    _predict_feature_types,
)


@pytest.fixture
def sample_feature() -> Feature:  # noqa: D103
    return Feature(name='name', column_name='col', description='desc', feature_type=FeatureType.NOMINAL)


@pytest.fixture
def sample_model_metadata(sample_feature) -> ModelMetadata:  # noqa: D103
    return ModelMetadata(model_name='my_model', features=[sample_feature])


def test_feature_creation_sets_properties_correctly():  # noqa: D103
    sut = Feature(name='name', column_name='col', description='desc', feature_type=FeatureType.NOMINAL)
    assert sut.name == 'name'
    assert sut.column_name == 'col'
    assert sut.description == 'desc'
    assert sut.feature_type == FeatureType.NOMINAL


# TODO: rewrite this using regexes
def test_feature_string_representation_contains_all_properties(sample_feature):  # noqa: D103
    sut = str(sample_feature)
    assert "Feature: name" in sut
    assert 'Column name' in sut
    assert 'Description' in sut
    assert 'Type' in sut


def test_model_metadata_creation_with_defaults_has_correct_properties():  # noqa: D103
    sut = ModelMetadata(model_name='model')
    assert sut.name == 'model'
    assert sut.model_purpose is None
    assert sut.model_problem == 'binary_classification'
    assert sut.features is not None
    assert len(sut.features) == 0
    assert sut.identifier_column_name == 'id'
    assert sut.prediction_column_name == 'p'
    assert sut.ground_truth_column_name == 'target'
    assert sut.partition_column_name == 'partition'
    assert sut.timestamp_column_name == 'date'


def test_model_metadata_creation_with_custom_values_has_correct_properties(sample_feature):  # noqa: D103
    sut = ModelMetadata(
        model_name='model',
        model_purpose='purpose',
        model_problem='classification',
        features=[sample_feature],
        identifier_column_name='ident',
        prediction_column_name='pred',
        ground_truth_column_name='gt',
        partition_column_name='part',
        timestamp_column_name='ts',
    )
    assert sut.name == 'model'
    assert sut.model_purpose == 'purpose'
    assert sut.model_problem == 'classification'
    assert len(sut.features) == 1
    assert sut.features[0].name == 'name'
    assert sut.features[0].column_name == 'col'
    assert sut.features[0].description == 'desc'
    assert sut.features[0].feature_type == FeatureType.NOMINAL
    assert sut.identifier_column_name == 'ident'
    assert sut.prediction_column_name == 'pred'
    assert sut.ground_truth_column_name == 'gt'
    assert sut.partition_column_name == 'part'
    assert sut.timestamp_column_name == 'ts'


# TODO: fix regexes
def test_model_metadata_string_representation_contains_all_properties(sample_model_metadata):  # noqa: D103
    sut = str(sample_model_metadata)
    assert re.match(rf"Metadata for model\s*{sample_model_metadata.name}", sut)
    # assert re.match(rf"Model purpose\s*.{'~ UNKNOWN ~'}.*", sut)
    # assert re.match(rf"Model problem\s*{sample_model_metadata.model_purpose or '~ UNKNOWN ~'}", sut)
    # assert re.match(rf"Identifier column\s*{sample_model_metadata.identifier_column_name or '~ UNKNOWN ~'}", sut)
    # assert re.match(rf"Timestamp column\s*{sample_model_metadata.timestamp_column_name or '~ UNKNOWN ~'}", sut)
    # assert re.match(rf"Partition column\s*{sample_model_metadata.partition_column_name or '~ UNKNOWN ~'}", sut)
    # assert re.match(rf"Ground truth column\s*{sample_model_metadata.ground_truth_column_name or '~ UNKNOWN ~'}", sut)

    # f = sample_model_metadata.features[0]
    # assert re.match(
    #     rf"Name\s*{f.name} Column\s*{f.column_name} Type\s*{f.feature_type} Description\s*{f.description}", sut)


def test_feature_filtering_by_index_delivers_correct_result(sample_model_metadata):  # noqa: D103
    features = [
        Feature(name=str.upper(c), column_name=c, feature_type=FeatureType.NOMINAL, description='')
        for c in ['a', 'b', 'c']
    ]
    sample_model_metadata.features = features
    for i in range(len(features)):
        assert sample_model_metadata.feature(index=i) == features[i]


def test_feature_filtering_by_index_with_out_of_bounds_index_raises_exception(sample_model_metadata):  # noqa: D103
    features = [
        Feature(name=str.upper(c), column_name=c, feature_type=FeatureType.NOMINAL, description='')
        for c in ['a', 'b', 'c']
    ]
    sample_model_metadata.features = features
    with pytest.raises(IndexError):
        _ = sample_model_metadata.feature(index=99)


def test_feature_filtering_by_feature_name_delivers_correct_result(sample_model_metadata):  # noqa: D103
    features = [
        Feature(name=str.upper(c), column_name=c, feature_type=FeatureType.NOMINAL, description='')
        for c in ['a', 'b', 'c']
    ]
    sample_model_metadata.features = features
    for i, c in enumerate(['a', 'b', 'c']):
        assert sample_model_metadata.feature(feature=str.upper(c)) == features[i]


def test_feature_filtering_by_feature_name_without_matches_returns_none(sample_model_metadata):  # noqa: D103
    features = [
        Feature(name=str.upper(c), column_name=c, feature_type=FeatureType.NOMINAL, description='')
        for c in ['a', 'b', 'c']
    ]
    sample_model_metadata.features = features
    assert sample_model_metadata.feature(feature='I dont exist') is None


def test_feature_filtering_by_column_name_returns_correct_result(sample_model_metadata):  # noqa: D103
    features = [
        Feature(name=str.upper(c), column_name=c, feature_type=FeatureType.NOMINAL, description='')
        for c in ['a', 'b', 'c']
    ]
    sample_model_metadata.features = features
    for i, c in enumerate(['a', 'b', 'c']):
        assert sample_model_metadata.feature(column=c) == features[i]


def test_feature_filtering_by_column_name_without_matches_returns_none(sample_model_metadata):  # noqa: D103
    features = [
        Feature(name=str.upper(c), column_name=c, feature_type=FeatureType.NOMINAL, description='')
        for c in ['a', 'b', 'c']
    ]
    sample_model_metadata.features = features
    assert sample_model_metadata.feature(column='I dont exist') is None


def test_feature_filtering_without_criteria_returns_none(sample_model_metadata):  # noqa: D103
    features = [
        Feature(name=str.upper(c), column_name=c, feature_type=FeatureType.NOMINAL, description='')
        for c in ['a', 'b', 'c']
    ]
    sample_model_metadata.features = features
    assert sample_model_metadata.feature() is None


def test_extract_metadata_for_no_cols_dataframe_should_return_none():  # noqa: D103
    sut = ModelMetadata(model_name='model').extract_metadata(data=pd.DataFrame())
    assert sut is None


def test_extract_metadata_without_any_feature_columns_should_return_metadata_without_features():  # noqa: D103
    data = pd.DataFrame(columns=['identity', 'prediction', 'actual', 'partition', 'ts'])
    sut = ModelMetadata('model').extract_metadata(data)
    assert len(sut.features) == 0


def test_extract_metadata_for_empty_dataframe_should_return_correct_column_names(sample_model_metadata):  # noqa: D103
    data = pd.DataFrame(columns=['identity', 'prediction', 'actual', 'partition', 'ts', 'feat1', 'feat2'])
    sut = ModelMetadata('model').extract_metadata(data)
    assert sut is not None
    assert sut.identifier_column_name == 'identity'
    assert sut.prediction_column_name == 'prediction'
    assert sut.ground_truth_column_name == 'actual'
    assert sut.partition_column_name == 'partition'
    assert sut.timestamp_column_name == 'ts'


# TODO verify behaviour
def test_extract_metadata_for_empty_dataframe_should_return_features_with_feature_type_unknown():  # noqa: D103
    data = pd.DataFrame(columns=['identity', 'prediction', 'actual', 'partition', 'ts', 'feat1', 'feat2'])
    sut = ModelMetadata('model').extract_metadata(data)
    assert len(sut.features) == 2
    assert sut.features[0].feature_type is FeatureType.UNKNOWN
    assert sut.features[1].feature_type is FeatureType.UNKNOWN


def test_extract_metadata_without_matching_columns_should_set_them_to_none():  # noqa: D103
    data = pd.DataFrame(columns=['a', 'b', 'c'])
    sut = ModelMetadata('model').extract_metadata(data, add_metadata=False)
    assert sut.identifier_column_name is None
    assert sut.prediction_column_name is None
    assert sut.ground_truth_column_name is None
    assert sut.partition_column_name is None
    assert sut.timestamp_column_name is None


def test_extract_metadata_without_matching_columns_should_set_features():  # noqa: D103
    data = pd.DataFrame(columns=['a', 'b', 'c'])
    sut = ModelMetadata('model').extract_metadata(data, add_metadata=False)
    assert len(sut.features) == 3
    assert sut.feature(column='a')
    assert sut.feature(column='b')
    assert sut.feature(column='c')


def test_extract_metadata_with_multiple_matching_columns_should_return_first_matching_column():  # noqa: D103
    data = pd.DataFrame(columns=['ident', 'id', 'uid'])
    sut = ModelMetadata('model').extract_metadata(data, add_metadata=False)
    assert sut.identifier_column_name == 'ident'


def test_extract_metadata_adds_metadata_columns_to_original_data_frame_by_default():  # noqa: D103
    cols = ['identity', 'prediction', 'actual', 'partition', 'ts', 'feat1', 'feat2']
    sut = pd.DataFrame(columns=cols)
    _ = ModelMetadata('model').extract_metadata(sut)
    assert len(sut.columns) == len(cols) + 5
    assert NML_METADATA_IDENTIFIER_COLUMN_NAME in sut.columns
    assert NML_METADATA_TIMESTAMP_COLUMN_NAME in sut.columns
    assert NML_METADATA_PARTITION_COLUMN_NAME in sut.columns
    assert NML_METADATA_PREDICTION_COLUMN_NAME in sut.columns
    assert NML_METADATA_GROUND_TRUTH_COLUMN_NAME in sut.columns


def test_extract_metadata_does_not_add_metadata_columns_when_add_metadata_parameter_is_false():  # noqa: D103
    cols = ['identity', 'prediction', 'actual', 'partition', 'ts', 'feat1', 'feat2']
    sut = pd.DataFrame(columns=cols)
    _ = ModelMetadata('model').extract_metadata(sut, add_metadata=False)
    assert len(sut.columns) == len(cols)
    assert NML_METADATA_IDENTIFIER_COLUMN_NAME not in sut.columns
    assert NML_METADATA_TIMESTAMP_COLUMN_NAME not in sut.columns
    assert NML_METADATA_PARTITION_COLUMN_NAME not in sut.columns
    assert NML_METADATA_PREDICTION_COLUMN_NAME not in sut.columns
    assert NML_METADATA_GROUND_TRUTH_COLUMN_NAME not in sut.columns


def test_extract_metadata_on_same_data_twice_does_not_add_duplicate_columns():  # noqa: D103
    cols = ['identity', 'prediction', 'actual', 'partition', 'ts', 'feat1', 'feat2']
    sut = pd.DataFrame(columns=cols)
    _ = ModelMetadata('model').extract_metadata(sut).extract_metadata(sut)
    assert len(sut.columns) == len(cols) + 5


@pytest.mark.parametrize(
    'col,expected',
    [
        ('id', True),
        ('ident', True),
        ('identity', True),
        ('identifier', True),
        ('uid', True),
        ('uuid', True),
        ('nope', False),
    ],
)
def test_guess_identifiers_yields_correct_results(col, expected):  # noqa: D103
    sut = _guess_identifiers(data=pd.DataFrame(columns=[col]))
    assert col == sut[0] if expected else len(sut) == 0


@pytest.mark.parametrize(
    'col,expected', [('date', True), ('timestamp', True), ('ts', True), ('date', True), ('time', True), ('nope', False)]
)
def test_guess_timestamps_yields_correct_results(col, expected):  # noqa: D103
    sut = _guess_timestamps(data=pd.DataFrame(columns=[col]))
    assert col == sut[0] if expected else len(sut) == 0


@pytest.mark.parametrize(
    'col,expected',
    [('p', True), ('pred', True), ('prediction', True), ('out', True), ('output', True), ('nope', False)],
)
def test_guess_predictions_yields_correct_results(col, expected):  # noqa: D103
    sut = _guess_predictions(data=pd.DataFrame(columns=[col]))
    assert col == sut[0] if expected else len(sut) == 0


@pytest.mark.parametrize(
    'col,expected', [('target', True), ('ground_truth', True), ('actual', True), ('actuals', True), ('nope', False)]
)
def test_guess_ground_truths_yields_correct_results(col, expected):  # noqa: D103
    sut = _guess_ground_truths(data=pd.DataFrame(columns=[col]))
    assert col == sut[0] if expected else len(sut) == 0


@pytest.mark.parametrize(
    'col,expected', [('part', False), ('partition', True), ('data_partition', True), ('nope', False)]
)
def test_guess_partitions_yields_correct_results(col, expected):  # noqa: D103
    sut = _guess_partitions(data=pd.DataFrame(columns=[col]))
    assert col == sut[0] if expected else len(sut) == 0


@pytest.mark.parametrize(
    'col,expected', [('part', True), ('A', True), ('partition', False), ('id', False), ('nope', True)]
)
def test_guess_features_yields_correct_results(col, expected):  # noqa: D103
    sut = _guess_features(data=pd.DataFrame(columns=[col]))
    assert col == sut[0] if expected else len(sut) == 0


def test_feature_type_detection_with_rows_under_num_rows_threshold_should_return_none():  # noqa: D103
    data = pd.DataFrame(columns=['a', 'b', 'c', 'd'])
    sut = _predict_feature_types(data)
    assert sut['predicted_feature_type'].map(lambda t: t == FeatureType.UNKNOWN).all()


def test_feature_type_detection_sets_float_cols_to_continuous():  # noqa: D103
    data = pd.DataFrame({'A': [math.pi for i in range(1000)]})
    sut = _predict_feature_types(data)
    assert sut.loc['A', 'predicted_feature_type'] == FeatureType.CONTINUOUS


def test_feature_type_detection_sets_above_high_cardinality_threshold_to_nominal():  # noqa: D103
    data = pd.DataFrame({'A': np.random.randint(75, size=100)})
    sut = _predict_feature_types(data)
    assert sut.loc['A', 'predicted_feature_type'] == FeatureType.CONTINUOUS


def test_feature_type_detection_sets_between_mid_and_high_cardinality_threshold_to_none():  # noqa: D103
    data = pd.DataFrame({'A': np.random.randint(50, size=1000)})
    sut = _predict_feature_types(data)
    assert sut.loc['A', 'predicted_feature_type'] == FeatureType.UNKNOWN


def test_feature_type_detection_sets_between_low_and_mid_cardinality_threshold_to_nominal():  # noqa: D103
    data = pd.DataFrame({'A': np.random.randint(6, size=1000)})
    sut = _predict_feature_types(data)
    assert sut.loc['A', 'predicted_feature_type'] == FeatureType.NOMINAL


def test_enrich_copies_each_metadata_column_to_new_fixed_column():  # noqa: D103
    data = pd.DataFrame(columns=['identity', 'prediction', 'actual', 'partition', 'ts', 'feat1', 'feat2'])
    md = ModelMetadata('model').extract_metadata(data)
    sut = md.enrich(data).columns

    assert NML_METADATA_IDENTIFIER_COLUMN_NAME in sut
    assert NML_METADATA_TIMESTAMP_COLUMN_NAME in sut
    assert NML_METADATA_PREDICTION_COLUMN_NAME in sut
    assert NML_METADATA_GROUND_TRUTH_COLUMN_NAME in sut
    assert NML_METADATA_PARTITION_COLUMN_NAME in sut


def test_enrich_works_on_copy_of_data_by_default():  # noqa: D103
    data = pd.DataFrame(columns=['identity', 'prediction', 'actual', 'partition', 'ts', 'feat1', 'feat2'])
    md = ModelMetadata('model').extract_metadata(data, add_metadata=False)
    sut = md.enrich(data).columns

    assert len(sut) == len(data.columns) + 5

    assert NML_METADATA_IDENTIFIER_COLUMN_NAME in sut
    assert NML_METADATA_TIMESTAMP_COLUMN_NAME in sut
    assert NML_METADATA_PREDICTION_COLUMN_NAME in sut
    assert NML_METADATA_GROUND_TRUTH_COLUMN_NAME in sut
    assert NML_METADATA_PARTITION_COLUMN_NAME in sut
    assert 'identity' in sut
    assert 'prediction' in sut
    assert 'actual' in sut
    assert 'partition' in sut
    assert 'feat1' in sut
    assert 'feat2' in sut


def test_enrich_works_on_original_data_when_in_place_is_true():  # noqa: D103
    data = pd.DataFrame(columns=['identity', 'prediction', 'actual', 'partition', 'ts', 'feat1', 'feat2'])
    md = ModelMetadata('model').extract_metadata(data, add_metadata=False)
    sut = md.enrich(data, in_place=True).columns

    assert len(sut) == len(data.columns)

    assert NML_METADATA_IDENTIFIER_COLUMN_NAME in data
    assert NML_METADATA_TIMESTAMP_COLUMN_NAME in data
    assert NML_METADATA_PREDICTION_COLUMN_NAME in data
    assert NML_METADATA_GROUND_TRUTH_COLUMN_NAME in data
    assert NML_METADATA_PARTITION_COLUMN_NAME in data
    assert 'identity' in data
    assert 'prediction' in data
    assert 'actual' in data
    assert 'partition' in data
    assert 'feat1' in data
    assert 'feat2' in data


def test_categorical_features_returns_only_nominal_features(sample_model_metadata):  # noqa: D103
    sample_model_metadata.features = [
        Feature(name='f1', column_name='f1', feature_type=FeatureType.NOMINAL),
        Feature(name='f2', column_name='f2', feature_type=FeatureType.UNKNOWN),
        Feature(name='f3', column_name='f3', feature_type=FeatureType.NOMINAL),
        Feature(name='f4', column_name='f4', feature_type=FeatureType.CONTINUOUS),
        Feature(name='f5', column_name='f5', feature_type=FeatureType.NOMINAL),
    ]

    sut = sample_model_metadata.categorical_features
    assert len(sut) == 3
    assert [f.name for f in sut] == ['f1', 'f3', 'f5']


def test_continuous_features_returns_only_continuous_features(sample_model_metadata):  # noqa: D103
    sample_model_metadata.features = [
        Feature(name='f1', column_name='f1', feature_type=FeatureType.NOMINAL),
        Feature(name='f2', column_name='f2', feature_type=FeatureType.UNKNOWN),
        Feature(name='f3', column_name='f3', feature_type=FeatureType.NOMINAL),
        Feature(name='f4', column_name='f4', feature_type=FeatureType.CONTINUOUS),
        Feature(name='f5', column_name='f5', feature_type=FeatureType.NOMINAL),
    ]

    sut = sample_model_metadata.continuous_features
    assert len(sut) == 1
    assert sut[0].name == 'f4'
