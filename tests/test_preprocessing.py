#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit tests for the preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from nannyml.exceptions import MissingMetadataException
from nannyml.metadata import NML_METADATA_COLUMNS, extract_metadata
from nannyml.preprocessing import preprocess


@pytest.fixture
def sample_data() -> pd.DataFrame:  # noqa: D103
    data = pd.DataFrame(pd.date_range(start='1/6/2020', freq='10min', periods=20 * 1008), columns=['timestamp'])
    data['week'] = data.timestamp.dt.isocalendar().week - 1
    data['partition'] = 'reference'
    data.loc[data.week >= 11, ['partition']] = 'analysis'
    # data[NML_METADATA_PARTITION_COLUMN_NAME] = data['partition']  # simulate preprocessing
    np.random.seed(167)
    data['f1'] = np.random.randn(data.shape[0])
    data['f2'] = np.random.rand(data.shape[0])
    data['f3'] = np.random.randint(4, size=data.shape[0])
    data['f4'] = np.random.randint(20, size=data.shape[0])
    data['output'] = np.random.randint(2, size=data.shape[0])
    data['actual'] = np.random.randint(2, size=data.shape[0])

    # Rule 1b is the shifted feature, 75% 0 instead of 50%
    rule1a = {2: 0, 3: 1}
    rule1b = {2: 0, 3: 0}
    data.loc[data.week < 16, ['f3']] = data.loc[data.week < 16, ['f3']].replace(rule1a)
    data.loc[data.week >= 16, ['f3']] = data.loc[data.week >= 16, ['f3']].replace(rule1b)

    # Rule 2b is the shifted feature
    c1 = 'white'
    c2 = 'red'
    c3 = 'green'
    c4 = 'blue'

    rule2a = {
        0: c1,
        1: c1,
        2: c1,
        3: c1,
        4: c1,
        5: c2,
        6: c2,
        7: c2,
        8: c2,
        9: c2,
        10: c3,
        11: c3,
        12: c3,
        13: c3,
        14: c3,
        15: c4,
        16: c4,
        17: c4,
        18: c4,
        19: c4,
    }

    rule2b = {
        0: c1,
        1: c1,
        2: c1,
        3: c1,
        4: c1,
        5: c2,
        6: c2,
        7: c2,
        8: c2,
        9: c2,
        10: c3,
        11: c3,
        12: c3,
        13: c1,
        14: c1,
        15: c4,
        16: c4,
        17: c4,
        18: c1,
        19: c2,
    }

    data.loc[data.week < 16, ['f4']] = data.loc[data.week < 16, ['f4']].replace(rule2a)
    data.loc[data.week >= 16, ['f4']] = data.loc[data.week >= 16, ['f4']].replace(rule2b)

    data.loc[data.week >= 16, ['f1']] = data.loc[data.week >= 16, ['f1']] + 0.6
    data.loc[data.week >= 16, ['f2']] = np.sqrt(data.loc[data.week >= 16, ['f2']])
    data['id'] = data.index
    data.drop(columns=['week'], inplace=True)

    return data


@pytest.fixture
def sample_metadata(sample_data):  # noqa: D103
    return extract_metadata(sample_data, model_name='model')


def test_preprocess_raises_missing_metadata_exception_when_metadata_is_not_complete(  # noqa: D103
    sample_data, sample_metadata
):
    sample_metadata.partition_column_name = None
    sample_data.drop(columns=['partition'], inplace=True)

    with pytest.raises(MissingMetadataException):
        _ = preprocess(data=sample_data, model_metadata=sample_metadata)


def test_preprocess_adds_metadata_columns_to_result(sample_data, sample_metadata):  # noqa: D103
    sut = preprocess(sample_data, sample_metadata)
    for col in NML_METADATA_COLUMNS:
        assert col in sut.columns


def test_preprocess_should_raise_warning_when_predicted_probabilities_outside_of_bounds(  # noqa: D103
    sample_data, sample_metadata
):
    sample_data.loc[10, 'output'] = 5
    sample_metadata.predicted_probability_column_name = 'output'

    with pytest.warns(
        UserWarning,
        match="the predicted probabilities column 'output' contains "
        "values outside of the accepted \\[0, 1\\] interval",
    ):
        _ = preprocess(sample_data, sample_metadata)


def test_preprocess_should_raise_warning_when_predicted_probabilities_have_too_few_unique_values(  # noqa: D103
    sample_data, sample_metadata
):
    sample_metadata.predicted_probability_column_name = 'output'

    with pytest.warns(
        UserWarning, match="the predicted probabilities column 'output' contains fewer than 2 " "unique values."
    ):
        _ = preprocess(sample_data, sample_metadata)
