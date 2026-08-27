import numpy as np
import pandas as pd
import pytest

from nannyml import PerformanceCalculator
from nannyml.exceptions import InvalidArgumentsException


def _binary_data(repeats=6):
    block = pd.DataFrame(
        {
            'id': ['tn', 'fp', 'fn', 'tp'],
            'y_true': [0, 0, 1, 1],
            'y_pred': [0, 1, 0, 1],
            'y_pred_proba': [0.1, 0.8, 0.4, 0.9],
        }
    )
    result = pd.concat([block] * repeats, ignore_index=True)
    result['id'] = [f'{value}-{index}' for index, value in enumerate(result['id'])]
    return result


def test_binary_performance_calculator_exposes_selected_instance_contributions():
    reference = _binary_data()
    analysis = _binary_data(repeats=2)
    calculator = PerformanceCalculator(
        metrics=['accuracy', 'precision', 'roc_auc', 'average_precision', 'confusion_matrix'],
        y_true='y_true',
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        problem_type='classification_binary',
        chunk_size=4,
    ).fit(reference)

    ordinary = calculator.calculate(analysis).to_df()
    result = calculator.calculate(
        analysis,
        include_instance_metrics=['accuracy', 'roc_auc', 'confusion_matrix'],
        identifier_column_name='id',
    )
    instance = result.to_instance_df()

    pd.testing.assert_frame_equal(result.to_df(), ordinary)
    assert instance['id'].tolist() == analysis['id'].tolist()
    assert 'accuracy_contribution' in instance
    assert 'roc_auc_contribution' in instance
    assert 'precision_contribution' not in instance
    assert 'average_precision_contribution' not in instance
    assert {
        'true_positive_contribution',
        'true_negative_contribution',
        'false_positive_contribution',
        'false_negative_contribution',
    }.issubset(instance.columns)
    assert instance.groupby('chunk_index')['accuracy_contribution'].mean().tolist() == [0.5, 0.5]
    assert instance.groupby('chunk_index')['roc_auc_contribution'].mean().tolist() == [0.75, 0.75]
    filtered_instance = result.filter(period='analysis', metrics=['accuracy']).to_instance_df()
    assert [column for column in filtered_instance if column.endswith('_contribution')] == ['accuracy_contribution']


@pytest.mark.parametrize('normalization', [None, 'all', 'true', 'pred'])
def test_binary_confusion_matrix_contributions_sum_to_each_component(normalization):
    reference = _binary_data()
    analysis = _binary_data(repeats=1)
    calculator = PerformanceCalculator(
        metrics=['confusion_matrix'],
        y_true='y_true',
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        problem_type='classification_binary',
        normalize_confusion_matrix=normalization,
        chunk_size=4,
    ).fit(reference)

    result = calculator.calculate(analysis, include_instance_metrics=True)
    instance = result.to_instance_df()
    chunk = result.filter(period='analysis').to_df().iloc[0]

    for component in ['true_positive', 'true_negative', 'false_positive', 'false_negative']:
        assert instance[f'{component}_contribution'].sum() == pytest.approx(chunk[(component, 'value')])


def test_regression_performance_contributions_follow_metric_reducers():
    reference = pd.DataFrame({'y_true': np.tile([1.0, 2.0, 3.0, 4.0], 6), 'y_pred': np.tile([1.0, 1.0, 5.0, 4.0], 6)})
    analysis = reference.iloc[:4].copy()
    calculator = PerformanceCalculator(
        metrics=['mae', 'mse', 'rmse'],
        y_true='y_true',
        y_pred='y_pred',
        problem_type='regression',
        chunk_size=4,
    ).fit(reference)

    result = calculator.calculate(analysis, include_instance_metrics=True)
    instance = result.to_instance_df()

    assert instance['identifier'].tolist() == analysis.index.tolist()
    assert instance['mae_contribution'].tolist() == [0.0, 1.0, 2.0, 0.0]
    assert instance['mse_contribution'].tolist() == [0.0, 1.0, 4.0, 0.0]
    assert instance['rmse_contribution'].tolist() == [0.0, 1.0, 4.0, 0.0]
    chunk = result.filter(period='analysis').to_df()
    assert instance['mae_contribution'].mean() == chunk[('mae', 'value')].iloc[0]
    assert np.sqrt(instance['rmse_contribution'].mean()) == chunk[('rmse', 'value')].iloc[0]


def test_performance_calculator_validates_instance_metric_selection_and_identifier():
    data = _binary_data()
    calculator = PerformanceCalculator(
        metrics=['accuracy'],
        y_true='y_true',
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        problem_type='classification_binary',
        chunk_size=4,
    ).fit(data)

    with pytest.raises(InvalidArgumentsException, match="'precision'"):
        calculator.calculate(data.iloc[:4], include_instance_metrics=['precision'])
    with pytest.raises(InvalidArgumentsException, match="'missing'"):
        calculator.calculate(data.iloc[:4], include_instance_metrics=True, identifier_column_name='missing')


def test_multiclass_performance_exposes_macro_and_confusion_matrix_contributions():
    block = pd.DataFrame(
        {
            'y_true': ['a', 'b', 'c', 'a', 'b', 'c'],
            'y_pred': ['a', 'c', 'c', 'b', 'b', 'a'],
            'proba_a': [0.8, 0.1, 0.1, 0.3, 0.2, 0.6],
            'proba_b': [0.1, 0.2, 0.1, 0.6, 0.7, 0.2],
            'proba_c': [0.1, 0.7, 0.8, 0.1, 0.1, 0.2],
        }
    )
    reference = pd.concat([block] * 6, ignore_index=True)
    calculator = PerformanceCalculator(
        metrics=[
            'accuracy',
            'precision',
            'recall',
            'specificity',
            'f1',
            'roc_auc',
            'average_precision',
            'confusion_matrix',
        ],
        y_true='y_true',
        y_pred='y_pred',
        y_pred_proba={'a': 'proba_a', 'b': 'proba_b', 'c': 'proba_c'},
        problem_type='classification_multiclass',
        chunk_size=6,
    ).fit(reference)

    result = calculator.calculate(block, include_instance_metrics=True)
    instance = result.to_instance_df()
    chunk = result.filter(period='analysis').to_df().iloc[0]

    assert instance['accuracy_contribution'].mean() == pytest.approx(chunk[('accuracy', 'value')])
    for metric in ['precision', 'recall', 'specificity', 'f1', 'average_precision']:
        assert instance[f'{metric}_contribution'].sum() == pytest.approx(chunk[(metric, 'value')])
    assert instance['roc_auc_contribution'].mean() == pytest.approx(chunk[('roc_auc', 'value')])
    for true_class in ['a', 'b', 'c']:
        for predicted_class in ['a', 'b', 'c']:
            component = f'true_{true_class}_pred_{predicted_class}'
            assert instance[f'{component}_contribution'].sum() == pytest.approx(chunk[(component, 'value')])
