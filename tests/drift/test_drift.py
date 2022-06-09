#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  #
#  License: Apache Software License 2.0

#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Tests for Drift package."""

import numpy as np
import pandas as pd
import plotly.graph_objects
import pytest

from nannyml.chunk import CountBasedChunker, DefaultChunker, PeriodBasedChunker, SizeBasedChunker
from nannyml.drift import DriftCalculator
from nannyml.drift.base import DriftResult
from nannyml.drift.model_inputs.univariate.statistical.calculator import UnivariateStatisticalDriftCalculator
from nannyml.exceptions import InvalidArgumentsException, MissingMetadataException
from nannyml.metadata import extract_metadata
from nannyml.metadata.base import NML_METADATA_COLUMNS, FeatureType
from nannyml.preprocessing import preprocess


@pytest.fixture
def sample_drift_data() -> pd.DataFrame:  # noqa: D103
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
    data['y_pred_proba'] = np.random.rand(data.shape[0])
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
    data.drop(columns=['week'], inplace=True)

    return data


@pytest.fixture
def sample_drift_data_with_nans(sample_drift_data) -> pd.DataFrame:  # noqa: D103
    data = sample_drift_data.copy(deep=True)
    data['id'] = data.index
    nan_pick1 = set(data.id.sample(frac=0.11, random_state=13))
    nan_pick2 = set(data.id.sample(frac=0.11, random_state=14))
    data.loc[data.id.isin(nan_pick1), 'f1'] = np.NaN
    data.loc[data.id.isin(nan_pick2), 'f4'] = np.NaN
    data.drop(columns=['id'], inplace=True)
    return data


@pytest.fixture
def sample_drift_metadata(sample_drift_data):  # noqa: D103
    return extract_metadata(sample_drift_data, model_name='model', model_type='classification_binary')


class SimpleDriftResult(DriftResult):
    """Dummy DriftResult implementation."""

    def plot(self, *args, **kwargs) -> plotly.graph_objects.Figure:
        """Fake plot."""
        pass


class SimpleDriftCalculator(DriftCalculator):
    """Dummy DriftCalculator implementation that returns a DataFrame with the selected feature columns, no rows."""

    def fit(self, reference_data: pd.DataFrame) -> DriftCalculator:  # noqa: D102
        _ = preprocess(reference_data, self.model_metadata, reference=True)
        return self

    def calculate(  # noqa: D102
        self,
        data: pd.DataFrame,
    ) -> SimpleDriftResult:
        data = preprocess(data, self.model_metadata)
        features_and_metadata = NML_METADATA_COLUMNS + self.selected_features
        chunks = self.chunker.split(data, columns=features_and_metadata, minimum_chunk_size=500)
        df = chunks[0].data.drop(columns=NML_METADATA_COLUMNS)
        return SimpleDriftResult(
            analysis_data=chunks, drift_data=pd.DataFrame(columns=df.columns), model_metadata=self.model_metadata
        )


def test_base_drift_calculator_given_empty_reference_data_should_raise_invalid_args_exception(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    ref_data = pd.DataFrame(columns=sample_drift_data.columns)
    calc = SimpleDriftCalculator(sample_drift_metadata)
    with pytest.raises(InvalidArgumentsException):
        calc.fit(ref_data)


def test_base_drift_calculator_given_empty_analysis_data_should_raise_invalid_args_exception(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    calc = SimpleDriftCalculator(sample_drift_metadata, chunk_size=1000)
    with pytest.raises(InvalidArgumentsException):
        calc.calculate(data=pd.DataFrame(columns=sample_drift_data.columns))


def test_base_drift_calculator_given_empty_features_list_should_calculate_for_all_features(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc = SimpleDriftCalculator(sample_drift_metadata, chunk_size=1000).fit(ref_data)
    sut = calc.calculate(data=sample_drift_data)

    md = extract_metadata(sample_drift_data, model_name='model', model_type='classification_binary')
    assert len(sut.data.columns) == len(md.features)
    for f in md.features:
        assert f.column_name in sut.data.columns


def test_base_drift_calculator_given_non_empty_features_list_should_only_calculate_for_these_features(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc = SimpleDriftCalculator(sample_drift_metadata, features=['f1', 'f3'], chunk_size=1000).fit(ref_data)
    _ = calc.calculate(data=sample_drift_data)
    sut = calc.calculate(data=sample_drift_data)

    assert len(sut.data.columns) == 2
    assert 'f1' in sut.data.columns
    assert 'f3' in sut.data.columns


def test_base_drift_calculator_uses_size_based_chunker_when_given_chunk_size(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    class TestDriftCalculator(DriftCalculator):
        def fit(self, reference_data: pd.DataFrame) -> DriftCalculator:
            return self

        def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
            data = preprocess(data, self.model_metadata)
            features_and_metadata = NML_METADATA_COLUMNS + self.selected_features
            chunks = self.chunker.split(data, columns=features_and_metadata, minimum_chunk_size=500)
            chunk_keys = [c.key for c in chunks]
            return pd.DataFrame({'keys': chunk_keys})

    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc = TestDriftCalculator(sample_drift_metadata, chunk_size=1000).fit(ref_data)
    sut = calc.calculate(sample_drift_data)['keys']
    expected = [
        c.key
        for c in SizeBasedChunker(1000).split(sample_drift_metadata.enrich(sample_drift_data), minimum_chunk_size=1)
    ]

    assert len(expected) == len(sut)
    assert sorted(expected) == sorted(sut)


def test_base_drift_calculator_uses_count_based_chunker_when_given_chunk_number(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    class TestDriftCalculator(DriftCalculator):
        def fit(self, reference_data: pd.DataFrame) -> DriftCalculator:
            self._suggested_minimum_chunk_size = 50
            return self

        def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
            data = preprocess(data, self.model_metadata)
            features_and_metadata = NML_METADATA_COLUMNS + self.selected_features
            chunks = self.chunker.split(data, columns=features_and_metadata, minimum_chunk_size=500)
            chunk_keys = [c.key for c in chunks]
            return pd.DataFrame({'keys': chunk_keys})

    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc = TestDriftCalculator(sample_drift_metadata, chunk_number=100).fit(ref_data)
    sut = calc.calculate(sample_drift_data)['keys']

    assert 101 == len(sut)


def test_base_drift_calculator_uses_period_based_chunker_when_given_chunk_period(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    class TestDriftCalculator(DriftCalculator):
        def fit(self, reference_data: pd.DataFrame) -> DriftCalculator:
            return self

        def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
            data = preprocess(data, self.model_metadata)
            features_and_metadata = NML_METADATA_COLUMNS + self.selected_features
            chunks = self.chunker.split(data, columns=features_and_metadata, minimum_chunk_size=500)
            chunk_keys = [c.key for c in chunks]
            return pd.DataFrame({'keys': chunk_keys})

    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc = TestDriftCalculator(sample_drift_metadata, chunk_period='W').fit(ref_data)
    sut = calc.calculate(sample_drift_data)['keys']

    assert 20 == len(sut)


def test_base_drift_calculator_uses_default_chunker_when_no_chunker_specified(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    class TestDriftCalculator(DriftCalculator):
        def fit(self, reference_data: pd.DataFrame) -> DriftCalculator:
            return self

        def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
            data = preprocess(data, self.model_metadata)
            features_and_metadata = NML_METADATA_COLUMNS + self.selected_features
            chunks = self.chunker.split(data, columns=features_and_metadata, minimum_chunk_size=500)
            chunk_keys = [c.key for c in chunks]
            return pd.DataFrame({'keys': chunk_keys})

    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc = TestDriftCalculator(sample_drift_metadata).fit(ref_data)
    sut = calc.calculate(sample_drift_data)['keys']
    expected = [
        c.key for c in DefaultChunker().split(sample_drift_metadata.enrich(sample_drift_data), minimum_chunk_size=500)
    ]

    assert len(expected) == len(sut)
    assert sorted(expected) == sorted(sut)


@pytest.mark.parametrize(
    'chunker',
    [
        (PeriodBasedChunker(offset='W')),
        (PeriodBasedChunker(offset='M')),
        (SizeBasedChunker(chunk_size=1000)),
        CountBasedChunker(chunk_count=25),
    ],
    ids=['chunk_period_weekly', 'chunk_period_monthly', 'chunk_size_1000', 'chunk_count_25'],
)
def test_univariate_statistical_drift_calculator_should_return_a_row_for_each_analysis_chunk_key(  # noqa: D103
    sample_drift_data, sample_drift_metadata, chunker
):
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc = UnivariateStatisticalDriftCalculator(sample_drift_metadata, chunker=chunker).fit(ref_data)
    sut = calc.calculate(data=sample_drift_data)

    chunks = chunker.split(sample_drift_metadata.enrich(sample_drift_data))
    assert len(chunks) == sut.data.shape[0]
    chunk_keys = [c.key for c in chunks]
    assert 'key' in sut.data.columns
    assert sorted(chunk_keys) == sorted(sut.data['key'].values)


def test_univariate_statistical_drift_calculator_should_contain_chunk_details(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc = UnivariateStatisticalDriftCalculator(sample_drift_metadata, chunk_period='W').fit(ref_data)

    drift = calc.calculate(data=sample_drift_data)

    sut = drift.data.columns
    assert 'key' in sut
    assert 'start_index' in sut
    assert 'start_date' in sut
    assert 'end_index' in sut
    assert 'end_date' in sut
    assert 'partition' in sut


def test_univariate_statistical_drift_calculator_returns_stat_column_and_p_value_column_for_each_feature(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc = UnivariateStatisticalDriftCalculator(sample_drift_metadata, chunk_size=1000).fit(ref_data)
    sut = calc.calculate(data=sample_drift_data).data.columns

    for f in sample_drift_metadata.features:
        if f.feature_type == FeatureType.CONTINUOUS:
            assert f'{f.column_name}_dstat' in sut
        else:
            assert f'{f.column_name}_chi2' in sut
        assert f'{f.column_name}_p_value' in sut


def test_univariate_statistical_drift_calculator(sample_drift_data, sample_drift_metadata):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    analysis_data = sample_drift_data.loc[sample_drift_data['partition'] == 'analysis']
    calc = UnivariateStatisticalDriftCalculator(sample_drift_metadata, chunk_period='W').fit(ref_data)
    try:
        _ = calc.calculate(data=analysis_data)
    except Exception:
        pytest.fail()


def test_statistical_drift_calculator_deals_with_missing_class_labels(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    # rig the data by setting all f3-values in first analysis chunk to 0
    sample_drift_data.loc[10080:16000, 'f3'] = 0
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    analysis_data = sample_drift_data.loc[sample_drift_data['partition'] == 'analysis']
    calc = UnivariateStatisticalDriftCalculator(sample_drift_metadata, chunk_size=5000).fit(ref_data)
    results = calc.calculate(data=analysis_data)
    assert not np.isnan(results.data.loc[0, 'f3_chi2'])
    assert not np.isnan(results.data.loc[0, 'f3_p_value'])


def test_statistical_drift_calculator_raises_missing_metadata_exception_when_features_missing(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    sample_drift_metadata.features = None
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']

    with pytest.raises(MissingMetadataException, match='features'):
        UnivariateStatisticalDriftCalculator(sample_drift_metadata, chunk_size=5000).fit(ref_data)


def test_statistical_drift_calculator_runs_without_unnecessary_metadata_properties(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    analysis_data = sample_drift_data.loc[sample_drift_data['partition'] == 'analysis']

    sample_drift_metadata.target_column_name = None
    sample_drift_metadata.prediction_column_name = None
    sample_drift_metadata.predicted_probability_column_name = None

    try:
        calc = UnivariateStatisticalDriftCalculator(sample_drift_metadata, chunk_size=5000).fit(ref_data)
        calc.calculate(data=analysis_data)
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")
