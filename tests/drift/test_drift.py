#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  #
#  License: Apache Software License 2.0

"""Tests for Drift package."""
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects
import pytest

from nannyml._typing import Key, Result, Self
from nannyml.base import Abstract1DResult, AbstractCalculator
from nannyml.chunk import CountBasedChunker, DefaultChunker, PeriodBasedChunker, SizeBasedChunker
from nannyml.drift.multivariate.data_reconstruction import DataReconstructionDriftCalculator
from nannyml.drift.univariate import UnivariateDriftCalculator
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_estimation.confidence_based import CBPE


@pytest.fixture(scope="module")
def sample_drift_data() -> pd.DataFrame:  # noqa: D103
    data = pd.DataFrame(pd.date_range(start='1/6/2020', freq='10min', periods=20 * 1008), columns=['timestamp'])
    data['week'] = data.timestamp.dt.isocalendar().week - 1
    data['period'] = 'reference'
    data.loc[data.week >= 11, ['period']] = 'analysis'
    # data[NML_METADATA_PERIOD_COLUMN_NAME] = data['period']  # simulate preprocessing
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

    data['f3'] = data['f3'].astype("category")

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


@pytest.fixture(scope="module")
def univariate_drift_result(sample_drift_data) -> Result:
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    calc = UnivariateDriftCalculator(
        column_names=['f1', 'f2', 'f3', 'f4'],
        timestamp_column_name='timestamp',
        continuous_methods=['kolmogorov_smirnov', 'jensen_shannon'],
        categorical_methods=['chi2', 'jensen_shannon'],
    ).fit(ref_data)
    return calc.calculate(data=sample_drift_data)


class SimpleDriftResult(Abstract1DResult):
    """Dummy DriftResult implementation."""

    def plot(self, *args, **kwargs) -> plotly.graph_objects.Figure:
        """Fake plot."""
        return plotly.graph_objects.Figure()

    def keys(self) -> List[Key]:
        return []

    def _filter(self, period: str, metrics: Optional[List[str]] = None, *args, **kwargs) -> Self:
        return self


class SimpleDriftCalculator(AbstractCalculator):
    """Dummy DriftCalculator implementation that returns a DataFrame with the selected feature columns, no rows."""

    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs):  # noqa: D102
        return self

    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> Result:  # noqa: D102
        return SimpleDriftResult(results_data=data, calculator=self)


def test_base_drift_calculator_uses_size_based_chunker_when_given_chunk_size(  # noqa: D103
    sample_drift_data,
):
    calc = SimpleDriftCalculator(chunk_size=1000)
    assert isinstance(calc.chunker, SizeBasedChunker)
    assert calc.chunker.chunk_size == 1000


def test_base_drift_calculator_uses_count_based_chunker_when_given_chunk_number(sample_drift_data):  # noqa: D103
    calc = SimpleDriftCalculator(chunk_number=1000)
    assert isinstance(calc.chunker, CountBasedChunker)
    assert calc.chunker.chunk_number == 1000


def test_base_drift_calculator_uses_period_based_chunker_when_given_chunk_period(sample_drift_data):  # noqa: D103
    calc = SimpleDriftCalculator(chunk_period='W', timestamp_column_name='timestamp')
    assert isinstance(calc.chunker, PeriodBasedChunker)
    assert calc.chunker.offset == 'W'


def test_base_drift_calculator_uses_default_chunker_when_no_chunker_specified(sample_drift_data):  # noqa: D103
    calc = SimpleDriftCalculator()
    assert isinstance(calc.chunker, DefaultChunker)


@pytest.mark.parametrize('column_names, expected', [('f1', ['f1']), (['f1', 'f2'], ['f1', 'f2'])])
def test_univariate_drift_calculator_create_with_single_or_list_of_column_names(column_names, expected):
    calc = UnivariateDriftCalculator(
        column_names=column_names,
        timestamp_column_name='timestamp',
        continuous_methods=['kolmogorov_smirnov'],
        categorical_methods=['chi2'],
    )
    assert calc.column_names == expected


@pytest.mark.parametrize(
    'continuous_methods, expected',
    [
        ('wasserstein', ['wasserstein']),
        (['wasserstein', 'jensen_shannon'], ['wasserstein', 'jensen_shannon']),
        (None, ['jensen_shannon']),
    ],
)
def test_univariate_drift_calculator_create_with_single_or_list_of_continuous_methods(continuous_methods, expected):
    calc = UnivariateDriftCalculator(
        column_names=['f1'],
        timestamp_column_name='timestamp',
        continuous_methods=continuous_methods,
        categorical_methods=['chi2'],
    )
    assert calc.continuous_method_names == expected


@pytest.mark.parametrize(
    'categorical_methods, expected',
    [('chi2', ['chi2']), (['chi2', 'jensen_shannon'], ['chi2', 'jensen_shannon']), (None, ['jensen_shannon'])],
)
def test_univariate_drift_calculator_create_with_single_or_list_of_categorical_methods(categorical_methods, expected):
    calc = UnivariateDriftCalculator(
        column_names=['f1'],
        timestamp_column_name='timestamp',
        continuous_methods=['jensen_shannon'],
        categorical_methods=categorical_methods,
    )
    assert calc.categorical_method_names == expected


@pytest.mark.parametrize(
    'chunker',
    [
        (PeriodBasedChunker(offset='W', timestamp_column_name='timestamp')),
        (PeriodBasedChunker(offset='M', timestamp_column_name='timestamp')),
        (SizeBasedChunker(chunk_size=1000)),
        CountBasedChunker(chunk_number=25),
    ],
    ids=['chunk_period_weekly', 'chunk_period_monthly', 'chunk_size_1000', 'chunk_count_25'],
)
def test_univariate_drift_calculator_should_return_a_row_for_each_analysis_chunk_key(  # noqa: D103
    sample_drift_data, chunker
):
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    calc = UnivariateDriftCalculator(
        column_names=['f1', 'f2', 'f3', 'f4'],
        timestamp_column_name='timestamp',
        continuous_methods=['kolmogorov_smirnov'],
        categorical_methods=['chi2'],
        chunker=chunker,
    ).fit(ref_data)
    sut = calc.calculate(data=sample_drift_data).filter(period='analysis').data

    chunks = chunker.split(sample_drift_data)
    assert len(chunks) == sut.shape[0]
    chunk_keys = [c.key for c in chunks]
    assert ('chunk', 'chunk', 'key') in sut.columns
    assert sorted(chunk_keys) == sorted(sut[('chunk', 'chunk', 'key')].values)


def test_univariate_statistical_drift_calculator_should_contain_chunk_details(sample_drift_data):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    calc = UnivariateDriftCalculator(
        column_names=['f1', 'f2', 'f3', 'f4'],
        timestamp_column_name='timestamp',
        continuous_methods=['kolmogorov_smirnov'],
        categorical_methods=['chi2'],
    ).fit(ref_data)

    drift = calc.calculate(data=sample_drift_data)

    sut = drift.data.columns
    assert ('chunk', 'chunk', 'key') in sut
    assert ('chunk', 'chunk', 'start_index') in sut
    assert ('chunk', 'chunk', 'start_date') in sut
    assert ('chunk', 'chunk', 'end_index') in sut
    assert ('chunk', 'chunk', 'end_date') in sut


def test_univariate_statistical_drift_calculator_returns_stat_column_for_each_feature(  # noqa: D103
    sample_drift_data,
):
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    calc = UnivariateDriftCalculator(
        column_names=['f1', 'f2', 'f3', 'f4'],
        timestamp_column_name='timestamp',
        continuous_methods=['kolmogorov_smirnov'],
        categorical_methods=['chi2'],
    ).fit(ref_data)

    sut = calc.calculate(data=sample_drift_data).data.columns

    for f in ['f1', 'f2']:
        assert (f, 'kolmogorov_smirnov', 'value') in sut

    for f in ['f3', 'f4']:
        assert (f, 'chi2', 'value') in sut


def test_statistical_drift_calculator_deals_with_missing_class_labels(sample_drift_data):  # noqa: D103
    # rig the data by setting all f3-values in first analysis chunk to 0
    sample_drift_data.loc[10080:16000, 'f3'] = 0
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    analysis_data = sample_drift_data.loc[sample_drift_data['period'] == 'analysis']
    calc = UnivariateDriftCalculator(
        column_names=['f1', 'f2', 'f3', 'f4'],
        timestamp_column_name='timestamp',
        continuous_methods=['kolmogorov_smirnov'],
        categorical_methods=['chi2'],
    ).fit(ref_data)
    results = calc.calculate(data=analysis_data)

    assert not np.isnan(results.data.loc[0, ('f3', 'chi2', 'value')])


@pytest.mark.parametrize(
    'calculator_opts, expected',
    [
        (
            {'chunk_size': 5000},
            [0.004968, 0.004833, 0.01186, 0.242068],
        ),
        (
            {'chunk_size': 5000, 'timestamp_column_name': 'timestamp'},
            [0.004968, 0.004833, 0.01186, 0.242068],
        ),
        (
            {'chunk_number': 5},
            [0.00873, 0.007688, 0.015179, 0.06503, 0.253323],
        ),
        (
            {'chunk_number': 5, 'timestamp_column_name': 'timestamp'},
            [0.00873, 0.007688, 0.015179, 0.06503, 0.253323],
        ),
        (
            {'chunk_period': 'M', 'timestamp_column_name': 'timestamp'},
            [
                0.007547,
                0.007895,
                0.009354,
                0.090575,
                0.256093,
            ],
        ),
        (
            {},
            [
                0.011012,
                0.017163,
                0.015675,
                0.010813,
                0.016865,
                0.014683,
                0.018552,
                0.113889,
                0.254861,
                0.253075,
            ],
        ),
        (
            {'timestamp_column_name': 'timestamp'},
            [
                0.011012,
                0.017163,
                0.015675,
                0.010813,
                0.016865,
                0.014683,
                0.018552,
                0.113889,
                0.254861,
                0.253075,
            ],
        ),
    ],
    ids=[
        'size_based_without_timestamp',
        'size_based_with_timestamp',
        'count_based_without_timestamp',
        'count_based_with_timestamp',
        'period_based_with_timestamp',
        'default_without_timestamp',
        'default_with_timestamp',
    ],
)
def test_univariate_statistical_drift_calculator_works_with_chunker(
    sample_drift_data, calculator_opts, expected  # noqa: D103
):
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    calc = UnivariateDriftCalculator(
        column_names=['f1', 'f2', 'f3', 'f4'],
        continuous_methods=['kolmogorov_smirnov'],
        categorical_methods=['chi2'],
        **calculator_opts,
    ).fit(ref_data)
    result = calc.calculate(data=sample_drift_data).filter(period='analysis').data
    sut = result[('f1', 'kolmogorov_smirnov', 'value')].to_list()
    assert all(np.round(sut, 6) == expected)


def test_statistical_drift_calculator_raises_type_error_when_features_missing():  # noqa: D103
    with pytest.raises(TypeError, match='column_names'):
        UnivariateDriftCalculator(timestamp_column_name='timestamp')


def test_statistical_drift_calculator_given_empty_reference_data_should_raise_invalid_args_exception(  # noqa: D103
    sample_drift_data,
):
    ref_data = pd.DataFrame(columns=sample_drift_data.columns)
    calc = UnivariateDriftCalculator(
        column_names=['f1', 'f2', 'f3', 'f4'],
        timestamp_column_name='timestamp',
        continuous_methods=['kolmogorov_smirnov'],
        categorical_methods=['chi2'],
    )
    with pytest.raises(InvalidArgumentsException):
        calc.fit(ref_data)


def test_base_drift_calculator_given_empty_analysis_data_should_raise_invalid_args_exception(  # noqa: D103
    sample_drift_data,
):
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    calc = UnivariateDriftCalculator(
        column_names=['f1', 'f2', 'f3', 'f4'],
        timestamp_column_name='timestamp',
        continuous_methods=['kolmogorov_smirnov'],
        categorical_methods=['chi2'],
    ).fit(ref_data)
    with pytest.raises(InvalidArgumentsException):
        calc.calculate(data=pd.DataFrame(columns=sample_drift_data.columns))


def test_base_drift_calculator_given_non_empty_features_list_should_only_calculate_for_these_features(  # noqa: D103
    sample_drift_data,
):
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    ana_data = sample_drift_data.loc[sample_drift_data['period'] == 'analysis']

    calc = UnivariateDriftCalculator(
        column_names=['f1', 'f3'],
        timestamp_column_name='timestamp',
        continuous_methods=['kolmogorov_smirnov'],
        categorical_methods=['chi2'],
    ).fit(ref_data)
    res = calc.calculate(data=ana_data)
    sut = list(set(res.data.columns.get_level_values(level=0)))

    assert 'f2' not in sut
    assert 'f4' not in sut


# See https://github.com/NannyML/nannyml/issues/192
def test_univariate_drift_calculator_returns_distinct_but_consistent_results_when_reused(sample_drift_data):
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    sut = UnivariateDriftCalculator(
        column_names=['f1', 'f3'],
        timestamp_column_name='timestamp',
        continuous_methods=['kolmogorov_smirnov'],
        categorical_methods=['chi2'],
    )
    sut.fit(ref_data)
    result1 = sut.calculate(data=sample_drift_data)
    result2 = sut.calculate(data=sample_drift_data)

    assert result1 is not result2
    pd.testing.assert_frame_equal(result1.to_df(), result2.to_df())


# See https://github.com/NannyML/nannyml/issues/197
def test_univariate_drift_result_extension_attributes_should_be_in_place(univariate_drift_result):
    # Data model is set up so f1, f2 are continuous and f3, f4 are categorical
    assert univariate_drift_result.continuous_column_names == ['f1', 'f2']
    assert univariate_drift_result.categorical_column_names == ['f3', 'f4']
    assert univariate_drift_result.timestamp_column_name == 'timestamp'
    assert univariate_drift_result.continuous_method_names == ['kolmogorov_smirnov', 'jensen_shannon']
    assert univariate_drift_result.categorical_method_names == ['chi2', 'jensen_shannon']
    assert [m.column_name for m in univariate_drift_result.continuous_methods] == [
        'kolmogorov_smirnov',
        'jensen_shannon',
    ]
    assert [m.column_name for m in univariate_drift_result.categorical_methods] == ['chi2', 'jensen_shannon']
    assert sorted(m.column_name for m in univariate_drift_result.methods) == sorted(
        ('chi2', 'kolmogorov_smirnov', 'jensen_shannon', 'jensen_shannon')
    )


# See https://github.com/NannyML/nannyml/issues/197
def test_univariate_drift_result_filter_should_preserve_data_with_default_args(univariate_drift_result):
    filtered_result = univariate_drift_result.filter()
    pd.testing.assert_frame_equal(filtered_result.data, univariate_drift_result.data, check_like=True)
    assert filtered_result.continuous_column_names == univariate_drift_result.continuous_column_names
    assert filtered_result.categorical_column_names == univariate_drift_result.categorical_column_names
    assert filtered_result.timestamp_column_name == univariate_drift_result.timestamp_column_name
    assert filtered_result.continuous_method_names == univariate_drift_result.continuous_method_names
    assert filtered_result.categorical_method_names == univariate_drift_result.categorical_method_names
    assert filtered_result.continuous_methods == univariate_drift_result.continuous_methods
    assert filtered_result.categorical_methods == univariate_drift_result.categorical_methods


# See https://github.com/NannyML/nannyml/issues/197
def test_unvariate_drift_result_filter_metrics(univariate_drift_result):
    filtered_result = univariate_drift_result.filter(methods=['chi2'])
    metrics = tuple(set(metric for (_, metric, _) in filtered_result.data.columns if metric != 'chunk'))
    assert metrics == ('chi2',)
    assert filtered_result.data.shape[0] == univariate_drift_result.data.shape[0]

    assert filtered_result.continuous_column_names == []
    assert filtered_result.categorical_column_names == ['f3', 'f4']
    assert filtered_result.continuous_method_names == []
    assert filtered_result.categorical_method_names == ['chi2']
    assert [m.column_name for m in filtered_result.continuous_methods] == []
    assert [m.column_name for m in filtered_result.categorical_methods] == ['chi2']
    assert sorted(m.column_name for m in filtered_result.methods) == ['chi2']


# See https://github.com/NannyML/nannyml/issues/197
def test_unvariate_drift_result_filter_column_names(univariate_drift_result):
    filtered_result = univariate_drift_result.filter(column_names=['f1', 'f2'])
    columns = tuple(sorted(set(column for (column, _, _) in filtered_result.data.columns if column != 'chunk')))
    assert columns == ('f1', 'f2')
    assert filtered_result.data.shape[0] == univariate_drift_result.data.shape[0]

    assert filtered_result.continuous_column_names == ['f1', 'f2']
    assert filtered_result.categorical_column_names == []
    assert filtered_result.continuous_method_names == ['kolmogorov_smirnov', 'jensen_shannon']
    assert filtered_result.categorical_method_names == []
    assert [m.column_name for m in filtered_result.continuous_methods] == ['kolmogorov_smirnov', 'jensen_shannon']
    assert [m.column_name for m in filtered_result.categorical_methods] == []
    assert sorted(m.column_name for m in filtered_result.methods) == sorted(('kolmogorov_smirnov', 'jensen_shannon'))


# See https://github.com/NannyML/nannyml/issues/197
def test_univariate_drift_result_filter_period(univariate_drift_result):
    filtered_result = univariate_drift_result.filter(period='reference')
    ref_period = univariate_drift_result.data.loc[
        univariate_drift_result.data.loc[:, ('chunk', 'chunk', 'period')] == 'reference', :
    ]
    pd.testing.assert_frame_equal(filtered_result.data, ref_period, check_like=True)

    assert filtered_result.continuous_column_names == ['f1', 'f2']
    assert filtered_result.categorical_column_names == ['f3', 'f4']
    assert filtered_result.continuous_method_names == ['kolmogorov_smirnov', 'jensen_shannon']
    assert filtered_result.categorical_method_names == ['chi2', 'jensen_shannon']
    assert [m.column_name for m in filtered_result.continuous_methods] == ['kolmogorov_smirnov', 'jensen_shannon']
    assert [m.column_name for m in filtered_result.categorical_methods] == ['chi2', 'jensen_shannon']
    assert sorted(m.column_name for m in filtered_result.methods) == sorted(
        ('chi2', 'kolmogorov_smirnov', 'jensen_shannon', 'jensen_shannon')
    )


@pytest.mark.parametrize(
    'calc_args, plot_args',
    [
        (
            {'timestamp_column_name': 'timestamp'},
            {'kind': 'drift'},
        ),
        ({}, {'kind': 'drift'}),
    ],
    ids=[
        'univariate_drift_with_timestamp',
        'univariate_drift_without_timestamp',
    ],
)
def test_result_plots_raise_no_exceptions(sample_drift_data, calc_args, plot_args):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    ana_data = sample_drift_data.loc[sample_drift_data['period'] == 'analysis']

    calc = UnivariateDriftCalculator(
        column_names=['f1', 'f3'],
        continuous_methods=['kolmogorov_smirnov'],
        categorical_methods=['chi2'],
        **calc_args,
    ).fit(ref_data)
    sut = calc.calculate(data=ana_data)

    try:
        _ = sut.plot(**plot_args)
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


@pytest.mark.parametrize(
    'cont_methods, cat_methods',
    [
        (
            [],
            ['chi2'],
        ),
        (
            ['jensen_shannon'],
            ['jensen_shannon'],
        ),
        (
            [],
            ['l_infinity'],
        ),
        (
            ['kolmogorov_smirnov'],
            [],
        ),
        (['wasserstein'], []),
        (['hellinger'], ['hellinger']),
    ],
    ids=[
        'feature_drift_with_ks_and_chi2',
        'feature_drift_with_js_and_js',
        'feature_drift_with_none_and_l_infinity',
        'feature_drift_with_ks_and_none',
        'feature_drift_with_wasserstein_and_none',
        'feature_drift_with_hellinger_and_hellinger',
    ],
)
def test_calculator_with_diff_methods_raise_no_exceptions(sample_drift_data, cont_methods, cat_methods):  # noqa: D103
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    ana_data = sample_drift_data.loc[sample_drift_data['period'] == 'analysis']
    try:
        calc = UnivariateDriftCalculator(
            column_names=['f1', 'f3'],
            timestamp_column_name='timestamp',
            continuous_methods=cont_methods,
            categorical_methods=cat_methods,
        ).fit(ref_data)
        calc.calculate(ana_data)
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


def test_repeat_calculation_results_return_only_latest_calculation_results(sample_drift_data):
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    ana_data = sample_drift_data.loc[sample_drift_data['period'] == 'analysis']

    chunker = SizeBasedChunker(chunk_size=1000, timestamp_column_name='timestamp')
    analysis_chunks = chunker.split(ana_data)

    calc = UnivariateDriftCalculator(
        column_names=['f1', 'f3'],
        continuous_methods=['kolmogorov_smirnov'],
        categorical_methods=['chi2'],
        chunker=chunker,
        timestamp_column_name='timestamp',
    ).fit(ref_data)

    res0 = calc.calculate(analysis_chunks[0].data).filter(period='analysis')

    res1 = calc.calculate(analysis_chunks[1].data).filter(period='analysis')

    res2 = calc.calculate(analysis_chunks[2].data).filter(period='analysis')

    assert len(res0) == 1
    assert res0.data.loc[0, ('chunk', 'chunk', 'start_date')] == analysis_chunks[0].start_datetime
    assert res0.data.loc[0, ('chunk', 'chunk', 'end_date')] == analysis_chunks[0].end_datetime
    assert len(res1) == 1
    assert res1.data.loc[0, ('chunk', 'chunk', 'start_date')] == analysis_chunks[1].start_datetime
    assert res1.data.loc[0, ('chunk', 'chunk', 'end_date')] == analysis_chunks[1].end_datetime
    assert len(res2) == 1
    assert res2.data.loc[0, ('chunk', 'chunk', 'start_date')] == analysis_chunks[2].start_datetime
    assert res2.data.loc[0, ('chunk', 'chunk', 'end_date')] == analysis_chunks[2].end_datetime


def test_result_comparison_to_multivariate_drift_plots_raise_no_exceptions(sample_drift_data):
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    ana_data = sample_drift_data.loc[sample_drift_data['period'] == 'analysis']

    calc = DataReconstructionDriftCalculator(column_names=['f1', 'f2', 'f3', 'f4']).fit(ref_data)
    result = calc.calculate(ana_data)

    calc2 = UnivariateDriftCalculator(
        column_names=['f1', 'f3'],
        continuous_methods=['kolmogorov_smirnov'],
        categorical_methods=['chi2'],
        timestamp_column_name='timestamp',
    ).fit(ref_data)
    result2 = calc2.calculate(ana_data)

    try:
        _ = result.compare(result2).plot()
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


def test_result_comparison_to_cbpe_plots_raise_no_exceptions(sample_drift_data):
    ref_data = sample_drift_data.loc[sample_drift_data['period'] == 'reference']
    ana_data = sample_drift_data.loc[sample_drift_data['period'] == 'analysis']

    calc = UnivariateDriftCalculator(
        column_names=['f1', 'f2', 'f3', 'f4'],
        continuous_methods=['kolmogorov_smirnov'],
        categorical_methods=['chi2'],
        timestamp_column_name='timestamp',
    ).fit(ref_data)
    result = calc.calculate(ana_data)

    calc2 = CBPE(
        timestamp_column_name='timestamp',
        y_pred_proba='y_pred_proba',
        y_pred='output',
        y_true='actual',
        metrics=['roc_auc', 'f1'],
        problem_type='classification_binary',
    ).fit(ref_data)
    result2 = calc2.estimate(ana_data)

    try:
        _ = result.compare(result2).plot()
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")
