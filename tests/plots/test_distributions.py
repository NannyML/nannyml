"""Regression tests for continuous distribution reporting."""

import numpy as np
import pandas as pd
import pytest

from nannyml import UnivariateDriftCalculator
from nannyml.chunk import SizeBasedChunker
from nannyml.plots.components.joy_plot import calculate_chunk_distributions


@pytest.mark.parametrize('index', [range(40, 64), [f'row-{i}' for i in range(24)]])
def test_continuous_distributions_preserve_chunks_with_non_default_index(index):  # noqa: D103
    data = pd.Series(np.linspace(0, 3, 24), name='feature')
    indexed_data = data.copy()
    indexed_data.index = index
    periods = pd.Series('analysis', index=range(len(data)))
    chunker = SizeBasedChunker(chunk_size=8)

    expected = calculate_chunk_distributions(data, chunker, data_periods=periods)
    actual = calculate_chunk_distributions(indexed_data, chunker, data_periods=periods)

    pd.testing.assert_frame_equal(actual[['chunk_key', 'period']], expected[['chunk_key', 'period']])
    for actual_support, expected_support in zip(actual['kde_support'], expected['kde_support']):
        np.testing.assert_allclose(actual_support, expected_support)
    for actual_density, expected_density in zip(actual['kde_density'], expected['kde_density']):
        np.testing.assert_allclose(actual_density, expected_density)


@pytest.mark.parametrize('index', [range(40, 64), [f'row-{i}' for i in range(24)]])
def test_univariate_distribution_plot_accepts_non_default_index(index):  # noqa: D103
    reference = pd.DataFrame({'feature': np.linspace(0, 3, 24)})
    analysis = pd.DataFrame({'feature': np.linspace(1, 4, 24)}, index=index)
    original = analysis.copy(deep=True)
    calculator = UnivariateDriftCalculator(
        column_names=['feature'], continuous_methods=['kolmogorov_smirnov'], chunk_size=8
    ).fit(reference)

    result = calculator.calculate(analysis)
    figure = result.plot(kind='distribution')
    assert len(result.filter(period='analysis').chunk_keys) == 3
    assert any(trace.name == 'Analysis' for trace in figure.data)
    pd.testing.assert_frame_equal(analysis, original)


def test_continuous_distribution_periods_follow_rows_when_sorted_by_timestamp():  # noqa: D103
    data = pd.DataFrame(
        {
            'feature': np.linspace(0, 3, 16),
            'timestamp': pd.date_range('2024-01-01', periods=16, freq='h'),
            'period': ['reference'] * 8 + ['analysis'] * 8,
        }
    )
    shuffled = data.iloc[[15, 0, 14, 1, 13, 2, 12, 3, 11, 4, 10, 5, 9, 6, 8, 7]].reset_index(drop=True)
    chunker = SizeBasedChunker(chunk_size=8, timestamp_column_name='timestamp')

    actual = calculate_chunk_distributions(
        shuffled['feature'], chunker, timestamps=shuffled['timestamp'], data_periods=shuffled['period']
    )
    expected = calculate_chunk_distributions(
        data['feature'], chunker, timestamps=data['timestamp'], data_periods=data['period']
    )

    pd.testing.assert_frame_equal(actual[['chunk_key', 'period']], expected[['chunk_key', 'period']])
    for actual_density, expected_density in zip(actual['kde_density'], expected['kde_density']):
        np.testing.assert_allclose(actual_density, expected_density)


@pytest.mark.parametrize(
    'chunking', [{'chunk_size': 8}, {'chunk_number': 3}, {'chunk_period': 'D'}], ids=['size', 'count', 'period']
)
def test_univariate_distribution_plot_preserves_period_timestamp(chunking):  # noqa: D103
    data = pd.DataFrame(
        {
            'feature': np.linspace(0, 3, 24),
            'timestamp': pd.date_range('2024-01-01', periods=24, freq='3h'),
        },
        index=[f'row-{i}' for i in range(24)],
    ).iloc[np.random.default_rng(42).permutation(24)]
    renamed = data.rename(columns={'timestamp': 'period'})
    original = renamed.copy(deep=True)

    def plot(frame, timestamp_column_name):
        calculator = UnivariateDriftCalculator(
            column_names=['feature'],
            continuous_methods=['kolmogorov_smirnov'],
            timestamp_column_name=timestamp_column_name,
            **chunking,
        ).fit(frame)
        return calculator.calculate(frame).plot(kind='distribution')

    expected = plot(data, 'timestamp')
    actual = plot(renamed, 'period')

    assert len(actual.data) == len(expected.data)
    assert any(len(trace.y) > 0 for trace in actual.data if trace.y is not None)
    for actual_trace, expected_trace in zip(actual.data, expected.data):
        assert actual_trace.name == expected_trace.name
        np.testing.assert_array_equal(actual_trace.x, expected_trace.x)
        np.testing.assert_array_equal(actual_trace.y, expected_trace.y)
    pd.testing.assert_frame_equal(renamed, original)
