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
    sut = ModelMetadata('model').extract_metadata(data)
    assert sut.identifier_column_name is None
    assert sut.prediction_column_name is None
    assert sut.ground_truth_column_name is None
    assert sut.partition_column_name is None
    assert sut.timestamp_column_name is None


def test_extract_metadata_without_matching_columns_should_set_features():  # noqa: D103
    data = pd.DataFrame(columns=['a', 'b', 'c'])
    sut = ModelMetadata('model').extract_metadata(data)
    assert len(sut.features) == 3
    assert sut.feature(column='a')
    assert sut.feature(column='b')
    assert sut.feature(column='c')


def test_extract_metadata_with_multiple_matching_columns_should_return_first_matching_column():  # noqa: D103
    data = pd.DataFrame(columns=['ident', 'id', 'uid'])
    sut = ModelMetadata('model').extract_metadata(data)
    assert sut.identifier_column_name == 'ident'


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
