import numpy as np
import pandas as pd
import pytest

from nannyml import (
    MissingValuesCalculator,
    NumericalRangeCalculator,
    SummaryStatsAvgCalculator,
    SummaryStatsMedianCalculator,
    SummaryStatsRowCountCalculator,
    SummaryStatsStdCalculator,
    SummaryStatsSumCalculator,
    UnseenValuesCalculator,
)


@pytest.mark.parametrize(
    'calculator_class, reducer',
    [
        (SummaryStatsAvgCalculator, lambda values: values.mean()),
        (SummaryStatsSumCalculator, lambda values: values.sum()),
        (SummaryStatsMedianCalculator, lambda values: values.median()),
        (SummaryStatsStdCalculator, lambda values: np.sqrt(values.sum() / (values.count() - 1))),
    ],
)
def test_summary_stat_calculators_expose_raw_or_squared_deviation_contributions(calculator_class, reducer):
    reference = pd.DataFrame({'value': np.tile([1.0, 2.0, 3.0, 4.0], 6)})
    analysis = pd.DataFrame({'id': ['a', 'b', 'c', 'd'], 'value': [2.0, 4.0, 6.0, 8.0]})
    calculator = calculator_class(column_names=['value'], chunk_size=4).fit(reference)

    result = calculator.calculate(analysis, include_instance_metrics=True, identifier_column_name='id')
    instance = result.to_instance_df()

    assert instance['id'].tolist() == analysis['id'].tolist()
    contribution = instance['value_contribution']
    chunk_value = result.filter(period='analysis').to_df()[('value', 'value')].iloc[0]
    assert reducer(contribution) == pytest.approx(chunk_value)


def test_row_count_exposes_unit_contributions():
    reference = pd.DataFrame({'value': range(24)})
    analysis = pd.DataFrame({'value': range(4)})
    calculator = SummaryStatsRowCountCalculator(chunk_size=4).fit(reference)

    instance = calculator.calculate(analysis, include_instance_metrics=True).to_instance_df()

    assert instance['rows_count_contribution'].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_std_instance_contributions_preserve_chunk_failure_handling():
    reference = pd.DataFrame({'value': np.tile([1.0, 2.0, 3.0, 4.0], 6)})
    analysis = pd.DataFrame({'value': ['one', 'two', 'three', 'four']})
    calculator = SummaryStatsStdCalculator(column_names=['value'], chunk_size=4).fit(reference)

    ordinary = calculator.calculate(analysis).to_df()
    result = calculator.calculate(analysis, include_instance_metrics=True)

    pd.testing.assert_frame_equal(result.to_df(), ordinary)
    assert result.to_instance_df()['value_contribution'].isna().all()


@pytest.mark.parametrize(
    'calculator, reference, analysis, expected',
    [
        (
            MissingValuesCalculator(column_names=['value'], chunk_size=4),
            pd.DataFrame({'value': np.tile([1.0, 2.0, 3.0, 4.0], 6)}),
            pd.DataFrame({'value': [1.0, np.nan, 2.0, np.nan]}),
            [0.0, 1.0, 0.0, 1.0],
        ),
        (
            UnseenValuesCalculator(column_names=['value'], chunk_size=4),
            pd.DataFrame({'value': np.tile(['a', 'b', 'a', 'b'], 6)}),
            pd.DataFrame({'value': ['a', 'c', 'b', 'd']}),
            [0.0, 1.0, 0.0, 1.0],
        ),
        (
            NumericalRangeCalculator(column_names=['value'], chunk_size=4),
            pd.DataFrame({'value': np.tile([0.0, 1.0, 2.0, 3.0], 6)}),
            pd.DataFrame({'value': [-1.0, 1.0, 2.0, 4.0]}),
            [1.0, 0.0, 0.0, 1.0],
        ),
    ],
)
def test_data_quality_calculators_expose_indicator_contributions(calculator, reference, analysis, expected):
    calculator.fit(reference)

    instance = calculator.calculate(analysis, include_instance_metrics=True).to_instance_df()

    assert instance['value_contribution'].tolist() == expected
