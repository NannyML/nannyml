#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Tests for Drift package."""

import numpy as np
import pandas as pd
import plotly.graph_objects
import pytest
from sklearn.impute import SimpleImputer

from nannyml.chunk import CountBasedChunker, DefaultChunker, PeriodBasedChunker, SizeBasedChunker
from nannyml.drift import BaseDriftCalculator
from nannyml.drift.base import DriftResult
from nannyml.drift.model_inputs.multivariate.data_reconstruction.calculator import (
    DataReconstructionDriftCalculator,
    _minimum_chunk_size,
)
from nannyml.drift.model_inputs.univariate.statistical.calculator import UnivariateStatisticalDriftCalculator
from nannyml.exceptions import InvalidArgumentsException, MissingMetadataException
from nannyml.metadata import NML_METADATA_COLUMNS, FeatureType, extract_metadata


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
    data['id'] = data.index
    data.drop(columns=['week'], inplace=True)

    return data


@pytest.fixture
def sample_drift_data_with_nans(sample_drift_data) -> pd.DataFrame:  # noqa: D103
    data = sample_drift_data.copy(deep=True)
    nan_pick1 = set(sample_drift_data.id.sample(frac=0.11, random_state=13))
    nan_pick2 = set(sample_drift_data.id.sample(frac=0.11, random_state=14))
    data.loc[data.id.isin(nan_pick1), 'f1'] = np.NaN
    data.loc[data.id.isin(nan_pick2), 'f4'] = np.NaN
    return data


@pytest.fixture
def sample_drift_metadata(sample_drift_data):  # noqa: D103
    return extract_metadata(sample_drift_data, model_name='model')


class SimpleDriftResult(DriftResult):
    """Dummy DriftResult implementation."""

    def plot(self, *args, **kwargs) -> plotly.graph_objects.Figure:
        """Fake plot."""
        pass


class SimpleDriftCalculator(BaseDriftCalculator):
    """Dummy DriftCalculator implementation that returns a DataFrame with the selected feature columns, no rows."""

    def _fit(self, reference_data: pd.DataFrame):
        pass

    def _calculate_drift(
        self,
        data: pd.DataFrame,
    ) -> SimpleDriftResult:
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
    calc = SimpleDriftCalculator(sample_drift_metadata, chunk_size=1000)
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc.fit(ref_data)
    sut = calc.calculate(data=sample_drift_data)

    md = extract_metadata(sample_drift_data, model_name='model')
    assert len(sut.data.columns) == len(md.features)
    for f in md.features:
        assert f.column_name in sut.data.columns


def test_base_drift_calculator_given_non_empty_features_list_should_only_calculate_for_these_features(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    calc = SimpleDriftCalculator(sample_drift_metadata, features=['f1', 'f3'], chunk_size=1000)
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc.fit(ref_data)
    _ = calc.calculate(data=sample_drift_data)
    sut = calc.calculate(data=sample_drift_data)

    assert len(sut.data.columns) == 2
    assert 'f1' in sut.data.columns
    assert 'f3' in sut.data.columns


def test_base_drift_calculator_uses_size_based_chunker_when_given_chunk_size(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    class TestDriftCalculator(BaseDriftCalculator):
        def _fit(self, reference_data: pd.DataFrame):
            pass

        def _calculate_drift(self, data: pd.DataFrame) -> pd.DataFrame:
            features_and_metadata = NML_METADATA_COLUMNS + self.selected_features
            chunks = self.chunker.split(data, columns=features_and_metadata, minimum_chunk_size=500)
            chunk_keys = [c.key for c in chunks]
            return pd.DataFrame({'keys': chunk_keys})

    calc = TestDriftCalculator(sample_drift_metadata, chunk_size=1000)
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc.fit(ref_data)
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
    class TestDriftCalculator(BaseDriftCalculator):
        def _fit(self, reference_data: pd.DataFrame):
            self._suggested_minimum_chunk_size = 50
            pass

        def _calculate_drift(self, data: pd.DataFrame) -> pd.DataFrame:
            features_and_metadata = NML_METADATA_COLUMNS + self.selected_features
            chunks = self.chunker.split(data, columns=features_and_metadata, minimum_chunk_size=500)
            chunk_keys = [c.key for c in chunks]
            return pd.DataFrame({'keys': chunk_keys})

    calc = TestDriftCalculator(sample_drift_metadata, chunk_number=100)
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc.fit(ref_data)
    sut = calc.calculate(sample_drift_data)['keys']

    assert 100 == len(sut)


def test_base_drift_calculator_uses_period_based_chunker_when_given_chunk_period(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    class TestDriftCalculator(BaseDriftCalculator):
        def _fit(self, reference_data: pd.DataFrame):
            pass

        def _calculate_drift(self, data: pd.DataFrame) -> pd.DataFrame:
            features_and_metadata = NML_METADATA_COLUMNS + self.selected_features
            chunks = self.chunker.split(data, columns=features_and_metadata, minimum_chunk_size=500)
            chunk_keys = [c.key for c in chunks]
            return pd.DataFrame({'keys': chunk_keys})

    calc = TestDriftCalculator(sample_drift_metadata, chunk_period='W')
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc.fit(ref_data)
    sut = calc.calculate(sample_drift_data)['keys']

    assert 20 == len(sut)


def test_base_drift_calculator_uses_default_chunker_when_no_chunker_specified(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    class TestDriftCalculator(BaseDriftCalculator):
        def _fit(self, reference_data: pd.DataFrame):
            pass

        def _calculate_drift(self, data: pd.DataFrame) -> pd.DataFrame:
            features_and_metadata = NML_METADATA_COLUMNS + self.selected_features
            chunks = self.chunker.split(data, columns=features_and_metadata, minimum_chunk_size=500)
            chunk_keys = [c.key for c in chunks]
            return pd.DataFrame({'keys': chunk_keys})

    calc = TestDriftCalculator(sample_drift_metadata)
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc.fit(ref_data)
    sut = calc.calculate(sample_drift_data)['keys']
    expected = [
        c.key for c in DefaultChunker().split(sample_drift_metadata.enrich(sample_drift_data), minimum_chunk_size=500)
    ]

    assert len(expected) == len(sut)
    assert sorted(expected) == sorted(sut)


def test_univariate_statistical_drift_calc_raises_missing_metadata_exception_when_predicted_proba_not_set(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    sample_drift_metadata.predicted_probability_column_name = None
    with pytest.raises(MissingMetadataException, match="missing value for 'predicted_probability_column_name'"):
        _ = UnivariateStatisticalDriftCalculator(sample_drift_metadata, chunk_size=5000)


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
    calc = UnivariateStatisticalDriftCalculator(sample_drift_metadata, chunker=chunker)
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc.fit(ref_data)
    sut = calc.calculate(data=sample_drift_data)

    chunks = chunker.split(sample_drift_metadata.enrich(sample_drift_data))
    assert len(chunks) == sut.data.shape[0]
    chunk_keys = [c.key for c in chunks]
    assert 'key' in sut.data.columns
    assert sorted(chunk_keys) == sorted(sut.data['key'].values)


def test_univariate_statistical_drift_calculator_should_contain_chunk_details(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    calc = UnivariateStatisticalDriftCalculator(sample_drift_metadata, chunk_period='W')
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc.fit(ref_data)

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
    calc = UnivariateStatisticalDriftCalculator(sample_drift_metadata, chunk_size=1000)
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc.fit(ref_data)
    sut = calc.calculate(data=sample_drift_data).data.columns

    for f in sample_drift_metadata.features:
        if f.feature_type == FeatureType.CONTINUOUS:
            assert f'{f.column_name}_dstat' in sut
        else:
            assert f'{f.column_name}_chi2' in sut
        assert f'{f.column_name}_p_value' in sut
    assert f'{sample_drift_metadata.predicted_probability_column_name}_dstat' in sut


def test_univariate_statistical_drift_calculator(sample_drift_data, sample_drift_metadata):  # noqa: D103
    calc = UnivariateStatisticalDriftCalculator(sample_drift_metadata, chunk_period='W')
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    analysis_data = sample_drift_data.loc[sample_drift_data['partition'] == 'analysis']
    calc.fit(ref_data)
    try:
        _ = calc.calculate(data=analysis_data)
    except Exception:
        pytest.fail()


def test_statistical_drift_calculator_deals_with_missing_class_labels(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    # rig the data by setting all f3-values in first analysis chunk to 0
    sample_drift_data.loc[10080:16000, 'f3'] = 0

    calc = UnivariateStatisticalDriftCalculator(sample_drift_metadata, chunk_size=5000)
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    analysis_data = sample_drift_data.loc[sample_drift_data['partition'] == 'analysis']
    calc.fit(ref_data)
    results = calc.calculate(data=analysis_data)
    assert not np.isnan(results.data.loc[0, 'f3_chi2'])
    assert not np.isnan(results.data.loc[0, 'f3_p_value'])


def test_data_reconstruction_drift_calculator_with_params_should_not_fail(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    calc = DataReconstructionDriftCalculator(sample_drift_metadata, n_components=0.75, chunk_period='W')
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc.fit(ref_data)
    try:
        drift = calc.calculate(data=sample_drift_data)
        print(drift)
    except Exception:
        pytest.fail()


def test_data_reconstruction_drift_calculator_with_default_params_should_not_fail(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    calc = DataReconstructionDriftCalculator(sample_drift_metadata, chunk_period='W')
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc.fit(ref_data)
    try:
        drift = calc.calculate(data=sample_drift_data)
        print(drift)
    except Exception:
        pytest.fail()


def test_data_reconstruction_drift_calculator_with_default_params_should_not_fail_w_nans(  # noqa: D103
    sample_drift_data_with_nans, sample_drift_metadata
):
    calc = DataReconstructionDriftCalculator(sample_drift_metadata, chunk_period='W')
    ref_data = sample_drift_data_with_nans.loc[sample_drift_data_with_nans['partition'] == 'reference']
    calc.fit(ref_data)
    try:
        drift = calc.calculate(data=sample_drift_data_with_nans)
        print(drift)
    except Exception:
        pytest.fail()


def test_data_reconstruction_drift_calculator_should_contain_chunk_details_and_single_drift_value_column(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    calc = DataReconstructionDriftCalculator(sample_drift_metadata, chunk_period='W')
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc.fit(ref_data)

    drift = calc.calculate(data=sample_drift_data)

    sut = drift.data.columns
    assert len(sut) == 10
    assert 'key' in sut
    assert 'start_index' in sut
    assert 'start_date' in sut
    assert 'end_index' in sut
    assert 'end_date' in sut
    assert 'partition' in sut
    assert 'upper_threshold' in sut
    assert 'lower_threshold' in sut
    assert 'alert' in sut
    assert 'reconstruction_error' in sut


def test_data_reconstruction_drift_calculator_should_contain_a_row_for_each_chunk(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    calc = DataReconstructionDriftCalculator(sample_drift_metadata, chunk_period='W')
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc.fit(ref_data)

    drift = calc.calculate(data=sample_drift_data)

    sample_drift_data = sample_drift_metadata.enrich(sample_drift_data)
    expected = len(PeriodBasedChunker(offset='W').split(sample_drift_data, minimum_chunk_size=1))
    sut = len(drift.data)
    assert sut == expected


# TODO: find a better way to test this
def test_data_reconstruction_drift_calculator_should_not_fail_when_using_feature_subset(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    calc = DataReconstructionDriftCalculator(
        model_metadata=sample_drift_metadata, features=['f1', 'f4'], chunk_period='W'
    )
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    try:
        calc.fit(ref_data)
        calc.calculate(sample_drift_data)
    except Exception as exc:
        pytest.fail(f"should not have failed but got {exc}")


def test_data_reconstruction_drift_calculator_numeric_results(sample_drift_data, sample_drift_metadata):  # noqa: D103
    calc = DataReconstructionDriftCalculator(sample_drift_metadata, chunk_period='W')
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc.fit(ref_data)
    drift = calc.calculate(data=sample_drift_data)
    expected_drift = pd.DataFrame.from_dict(
        {
            'key': [
                '2020-01-06/2020-01-12',
                '2020-01-13/2020-01-19',
                '2020-01-20/2020-01-26',
                '2020-01-27/2020-02-02',
                '2020-02-03/2020-02-09',
                '2020-02-10/2020-02-16',
                '2020-02-17/2020-02-23',
                '2020-02-24/2020-03-01',
                '2020-03-02/2020-03-08',
                '2020-03-09/2020-03-15',
                '2020-03-16/2020-03-22',
                '2020-03-23/2020-03-29',
                '2020-03-30/2020-04-05',
                '2020-04-06/2020-04-12',
                '2020-04-13/2020-04-19',
                '2020-04-20/2020-04-26',
                '2020-04-27/2020-05-03',
                '2020-05-04/2020-05-10',
                '2020-05-11/2020-05-17',
                '2020-05-18/2020-05-24',
            ],
            'reconstruction_error': [
                0.795939312162986,
                0.7840110463966236,
                0.8119098730091425,
                0.7982130082187159,
                0.807815521612754,
                0.8492042669464963,
                0.7814127409090083,
                0.8022621626300768,
                0.8104742129966831,
                0.7703901270625767,
                0.8007070128606296,
                0.7953169982962172,
                0.7862784182468701,
                0.838376989270861,
                0.8019280640410021,
                0.7154339372837247,
                0.7171169593894968,
                0.7255999561968017,
                0.73493013255886,
                0.7777717388501538,
            ],
        }
    )
    pd.testing.assert_frame_equal(expected_drift, drift.data[['key', 'reconstruction_error']])


def test_data_reconstruction_drift_calculator_with_only_numeric_should_not_fail(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    calc = DataReconstructionDriftCalculator(
        sample_drift_metadata,
        chunk_period='W',
        features=[el.column_name for el in sample_drift_metadata.continuous_features],
    )
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc.fit(ref_data)
    try:
        calc.calculate(data=sample_drift_data)
    except Exception:
        pytest.fail()


def test_data_reconstruction_drift_calculator_with_only_categorical_should_not_fail(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    calc = DataReconstructionDriftCalculator(
        sample_drift_metadata,
        chunk_period='W',
        features=[el.column_name for el in sample_drift_metadata.categorical_features],
    )
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc.fit(ref_data)
    try:
        calc.calculate(data=sample_drift_data)
    except Exception:
        pytest.fail()


def test_data_reconstruction_drift_calculator_minimum_chunk_size_yields_correct_result(  # noqa: D103
    sample_drift_data, sample_drift_metadata
):
    features = [el.column_name for el in sample_drift_metadata.features]
    calc = DataReconstructionDriftCalculator(
        sample_drift_metadata,
        chunk_period='W',
        features=features,
    )
    ref_data = sample_drift_data.loc[sample_drift_data['partition'] == 'reference']
    calc.fit(ref_data)
    assert _minimum_chunk_size(features) == 63


def test_data_reconstruction_drift_calculator_given_wrong_cat_imputer_object_raises_typeerror(  # noqa: D103
    sample_drift_data_with_nans, sample_drift_metadata
):
    with pytest.raises(TypeError):
        DataReconstructionDriftCalculator(
            model_metadata=sample_drift_metadata,
            chunk_period='W',
            imputer_categorical=5,
            imputer_continuous=SimpleImputer(missing_values=np.nan, strategy='mean'),
        )


def test_data_reconstruction_drift_calculator_given_wrong_cat_imputer_strategy_raises_valueerror(  # noqa: D103
    sample_drift_data_with_nans, sample_drift_metadata
):
    with pytest.raises(ValueError):
        DataReconstructionDriftCalculator(
            model_metadata=sample_drift_metadata,
            chunk_period='W',
            imputer_categorical=SimpleImputer(missing_values=np.nan, strategy='median'),
            imputer_continuous=SimpleImputer(missing_values=np.nan, strategy='mean'),
        )


def test_data_reconstruction_drift_calculator_given_wrong_cont_imputer_object_raises_typeerror(  # noqa: D103
    sample_drift_data_with_nans, sample_drift_metadata
):
    with pytest.raises(TypeError):
        DataReconstructionDriftCalculator(
            model_metadata=sample_drift_metadata,
            chunk_period='W',
            imputer_categorical=SimpleImputer(missing_values=np.nan, strategy='most_frequent'),
            imputer_continuous=5,
        )
