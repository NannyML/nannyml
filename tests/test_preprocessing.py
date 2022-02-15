#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit tests for the preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from nannyml.metadata import extract_metadata
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


def test_preprocess_logs_error_message_when_metadata_is_not_complete(  # noqa: D103
    sample_data, sample_metadata, caplog
):
    sample_metadata.partition_column_name = None
    sample_data.drop(columns=['partition'], inplace=True)

    _, _ = preprocess(data=sample_data, model_name='my_model')

    assert len(caplog.records) != 0
    sut = caplog.records[-1]
    assert sut.levelname == 'ERROR'
    assert sut.name == 'nannyml.preprocessing'
    assert 'metadata is still missing values' in sut.msg


def test_preprocess_returns_partial_metadata_when_metadata_is_not_complete(sample_data, sample_metadata):  # noqa: D103
    sample_data.drop(columns=['partition'], inplace=True)

    sut, _ = preprocess(data=sample_data, model_name='my_model')

    assert sut is not None
    assert sut.partition_column_name is None
    assert sut.identifier_column_name == sample_metadata.identifier_column_name
    assert sut.timestamp_column_name == sample_metadata.timestamp_column_name
    assert sut.prediction_column_name == sample_metadata.prediction_column_name
    assert sut.ground_truth_column_name == sample_metadata.ground_truth_column_name
    assert len(sut.features) == len(sample_metadata.features)


def test_preprocess_returns_empty_dataframe_when_metadata_is_not_complete(sample_data):  # noqa: D103
    sample_data.drop(columns=['partition'], inplace=True)

    _, sut = preprocess(data=sample_data, model_name='my_model')

    assert sut is None
