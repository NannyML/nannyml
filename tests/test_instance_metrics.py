import numpy as np
import pandas as pd
import pytest

from nannyml.base import Abstract1DResult, AbstractCalculator, AbstractEstimator
from nannyml.chunk import SizeBasedChunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.instance_metrics import (
    INSTANCE_IDENTIFIER_COLUMN_NAME,
    _soft_auc_contributions,
    build_instance_chunk_record,
    prepare_data_for_instance_metrics,
    resolve_instance_metric_selection,
)
from nannyml.performance_estimation.confidence_based.metrics import estimate_roc_auc


class _Result(Abstract1DResult):
    def keys(self):
        return []

    def plot(self, *args, **kwargs):
        raise NotImplementedError


class _UnsupportedCalculator(AbstractCalculator):
    def _fit(self, reference_data, *args, **kwargs):
        return self

    def _calculate(self, data, *args, **kwargs):
        return _Result(pd.DataFrame({('chunk', 'key'): ['chunk']}))


class _UnsupportedEstimator(AbstractEstimator):
    def _fit(self, reference_data, *args, **kwargs):
        return self

    def _estimate(self, data, *args, **kwargs):
        return _Result(pd.DataFrame({('chunk', 'key'): ['chunk']}))


def test_result_keeps_chunk_output_unchanged_and_exports_instance_data_separately():
    chunk_data = pd.DataFrame({('chunk', 'key'): ['[0:1]'], ('metric', 'value'): [0.5]})
    instance_data = pd.DataFrame(
        {
            'identifier': ['a', 'b'],
            'chunk_key': ['[0:1]', '[0:1]'],
            'chunk_index': [0, 0],
            'period': ['analysis', 'analysis'],
            'metric_contribution': [1.0, 0.0],
        }
    )

    result = _Result(chunk_data, instance_data=instance_data)

    pd.testing.assert_frame_equal(result.to_df(), chunk_data)
    pd.testing.assert_frame_equal(result.to_instance_df(), instance_data)
    exported = result.to_instance_df()
    exported.loc[0, 'metric_contribution'] = -1
    assert result.to_instance_df().loc[0, 'metric_contribution'] == 1


def test_result_explains_when_instance_metrics_were_not_requested():
    result = _Result(pd.DataFrame({('chunk', 'key'): ['[0:1]']}))

    with pytest.raises(InvalidArgumentsException, match='include_instance_metrics'):
        result.to_instance_df()


def test_result_keeps_extra_positional_arguments_backward_compatible():
    result = _Result(pd.DataFrame({('chunk', 'key'): ['[0:1]']}), object())

    assert result.instance_data is None


def test_base_interfaces_reject_instance_metrics_for_unsupported_algorithms():
    with pytest.raises(InvalidArgumentsException, match='does not support'):
        _UnsupportedCalculator().calculate(pd.DataFrame({'x': [1]}), include_instance_metrics=True)
    with pytest.raises(InvalidArgumentsException, match='does not support'):
        _UnsupportedEstimator().estimate(pd.DataFrame({'x': [1]}), include_instance_metrics=True)


@pytest.mark.parametrize(
    'requested, expected',
    [(False, []), (True, ['accuracy', 'f1']), (['f1'], ['f1'])],
)
def test_resolve_instance_metric_selection(requested, expected):
    assert resolve_instance_metric_selection(requested, ['accuracy', 'f1']) == expected


def test_resolve_instance_metric_selection_rejects_unknown_metrics():
    with pytest.raises(InvalidArgumentsException, match="'precision'"):
        resolve_instance_metric_selection(['precision'], ['accuracy', 'f1'])


def test_soft_auc_contributions_reconcile_tied_scores_to_existing_estimate():
    probabilities = np.asarray([0.9, 0.1, 0.8, 0.2])
    scores = np.asarray([0.5, 0.5, 0.1, 0.1])

    contributions = _soft_auc_contributions(probabilities, scores)

    assert contributions.mean() == pytest.approx(estimate_roc_auc(probabilities, scores))


def test_prepare_data_uses_index_by_default_and_requires_unique_identifiers():
    data = pd.DataFrame({'value': [1, 2]}, index=['row-1', 'row-2'])

    prepared, output_name = prepare_data_for_instance_metrics(data, None)

    assert output_name == 'identifier'
    assert prepared[INSTANCE_IDENTIFIER_COLUMN_NAME].tolist() == ['row-1', 'row-2']
    with pytest.raises(InvalidArgumentsException, match='unique'):
        prepare_data_for_instance_metrics(pd.DataFrame({'value': [1, 2]}, index=[0, 0]), None)


def test_prepare_data_uses_named_identifier_column_and_validates_it():
    data = pd.DataFrame({'id': ['a', 'b'], 'value': [1, 2]})

    prepared, output_name = prepare_data_for_instance_metrics(data, 'id')

    assert output_name == 'id'
    assert prepared[INSTANCE_IDENTIFIER_COLUMN_NAME].tolist() == ['a', 'b']
    with pytest.raises(InvalidArgumentsException, match="'missing'"):
        prepare_data_for_instance_metrics(data, 'missing')


def test_prepare_data_uses_unique_multiindex_tuple_identifiers():
    index = pd.MultiIndex.from_tuples([('customer-a', 1), ('customer-b', 2)])
    data = pd.DataFrame({'value': [1, 2]}, index=index)

    prepared, output_name = prepare_data_for_instance_metrics(data, None)

    assert output_name == 'identifier'
    assert prepared[INSTANCE_IDENTIFIER_COLUMN_NAME].tolist() == [('customer-a', 1), ('customer-b', 2)]


def test_prepare_data_validates_multiindex_identifiers():
    duplicate_index = pd.MultiIndex.from_tuples([('customer-a', 1), ('customer-a', 1)])
    missing_index = pd.MultiIndex.from_tuples([('customer-a', 1), (None, 2)])

    with pytest.raises(InvalidArgumentsException, match='unique'):
        prepare_data_for_instance_metrics(pd.DataFrame({'value': [1, 2]}, index=duplicate_index), None)
    with pytest.raises(InvalidArgumentsException, match='missing'):
        prepare_data_for_instance_metrics(pd.DataFrame({'value': [1, 2]}, index=missing_index), None)


@pytest.mark.parametrize('identifier_column_name', ['period', 'metric_contribution'])
def test_prepare_data_rejects_identifier_names_that_conflict_with_instance_output(identifier_column_name):
    data = pd.DataFrame({identifier_column_name: ['a', 'b'], 'value': [1, 2]})

    with pytest.raises(InvalidArgumentsException, match='cannot be used'):
        prepare_data_for_instance_metrics(data, identifier_column_name)


def test_build_instance_chunk_record_adds_only_identifiers_chunk_metadata_and_contributions():
    prepared, output_name = prepare_data_for_instance_metrics(
        pd.DataFrame({'id': ['a', 'b'], 'secret_feature': [10, 20]}), 'id'
    )
    chunk = SizeBasedChunker(chunk_size=2).split(prepared)[0]

    result = build_instance_chunk_record(
        chunk,
        identifier_column_name=output_name,
        period='analysis',
        contributions={'accuracy_contribution': pd.Series([1.0, 0.0], index=chunk.data.index)},
    )

    assert result.columns.tolist() == [
        'id',
        'chunk_key',
        'chunk_index',
        'chunk_start_index',
        'chunk_end_index',
        'chunk_start_date',
        'chunk_end_date',
        'period',
        'accuracy_contribution',
    ]
    assert 'secret_feature' not in result
