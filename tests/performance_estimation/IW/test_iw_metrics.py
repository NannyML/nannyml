import pandas as pd
import pytest

from nannyml.chunk import DefaultChunker
from nannyml.datasets import (
    load_synthetic_car_loan_dataset,
    load_synthetic_multiclass_classification_dataset,
)
from nannyml.performance_estimation.importance_weighting import IW
# from nannyml.performance_estimation.importance_weighting.metrics import (
#     BinaryClassificationAccuracy,
#     BinaryClassificationAUROC,
#     BinaryClassificationConfusionMatrix,
#     BinaryClassificationF1,
#     BinaryClassificationPrecision,
#     BinaryClassificationRecall,
#     BinaryClassificationSpecificity,
# )
# from nannyml.thresholds import ConstantThreshold


@pytest.mark.parametrize(
    'calculator_opts, expected',
    [
        (
            {
                'normalize_confusion_matrix': None,
                'business_value_matrix': [[2, -5], [-10, 10]],
                'normalize_business_value': None,
            },
            pd.DataFrame(
                {
                    'key': ['[0:4999]', '[5000:9999]', '[10000:14999]', '[0:4999]', '[5000:9999]'],
                    'estimated_roc_auc': [0.972845, 0.971718, 0.972491, 0.972685, 0.971584],
                    'estimated_f1': [0.941566, 0.940875, 0.945742, 0.937973, 0.938716],
                    'estimated_precision': [0.957673, 0.955758, 0.958106, 0.950811, 0.948012],
                    'estimated_recall': [0.925991, 0.926449, 0.933694, 0.925478, 0.9296],
                    'estimated_specificity': [0.960513, 0.958986, 0.957813, 0.952975, 0.94577],
                    'estimated_accuracy': [0.943561, 0.943081, 0.945556, 0.939350, 0.937436],
                    'estimated_true_positive': [10643.491309, 10659.439831, 11258.494863, 4954.564752, 5164.873846],
                    'estimated_true_negative': [11442.715807, 11537.400826, 11176.964196, 5194.346387, 4939.721549],
                    'estimated_false_positive': [470.413482, 493.427409, 492.291028, 256.318617, 283.239176],
                    'estimated_false_negative': [850.667061, 846.256637, 799.519332, 398.956322, 391.141989],
                    'estimated_business_value': [118461.606687, 118739.496543, 124482.228569, 54663.183996, 56200.565786],
                }
            ),
        ),
    ],
    ids=[
        'size_based_without_timestamp_cm_normalization_none_business_norm_none',
    ],
)
def test_iw_for_binary_classification_estimations(calculator_opts, expected):
    reference, analysis, _ = load_synthetic_car_loan_dataset()
    cbpe = IW(  # type: ignore
        feature_column_names=[
            "car_value",
            "debt_to_income_ratio",
            "loan_length",
            "driver_tenure",
            "salary_range",
            "repaid_loan_on_prev_car",
            "size_of_downpayment"
        ],
        y_true='repaid',
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        problem_type='classification_binary',
        metrics=[
            'roc_auc',
            'f1',
            'precision',
            'recall',
            'specificity',
            'accuracy',
            'confusion_matrix',
            'business_value',
        ],
        chunk_size=5_000,
        **calculator_opts,
    ).fit(reference.head(15_000))
    result = cbpe.estimate(analysis.tail(10_000))
    metric_column_names = [name for metric in result.metrics for name in metric.column_names]
    sut = result.to_df()[[('chunk', 'key')] + [(c, 'value') for c in metric_column_names]]
    sut.columns = [
        'key',
        'estimated_roc_auc',
        'estimated_f1',
        'estimated_precision',
        'estimated_recall',
        'estimated_specificity',
        'estimated_accuracy',
        'estimated_true_positive',
        'estimated_true_negative',
        'estimated_false_positive',
        'estimated_false_negative',
        'estimated_business_value',
    ]
    pd.testing.assert_frame_equal(expected, sut.round(6))


@pytest.mark.parametrize(
    'calculator_opts, expected',
    [
        (
            {'timestamp_column_name': 'timestamp', 'normalize_confusion_matrix': 'true'},
            pd.DataFrame(
                {
                    'key': ['[0:4999]', '[5000:9999]', '[10000:14999]', '[0:4999]', '[5000:9999]'],
                    'estimated_roc_auc': [0.906573, 0.907017, 0.907395, 0.796482, 0.791788],
                    'estimated_f1': [0.753016, 0.754558, 0.754614, 0.60351, 0.602515],
                    'estimated_precision': [0.75302, 0.75464, 0.754723, 0.606623, 0.607366],
                    'estimated_recall': [0.753013, 0.754499, 0.754536, 0.603631, 0.601305],
                    'estimated_specificity': [0.876473, 0.877146, 0.877239, 0.803273, 0.803623],
                    'estimated_accuracy': [0.752975, 0.754415, 0.754607, 0.610914, 0.614981],
                    'estimated_true_highstreet_card_pred_highstreet_card': [
                        0.768271, 0.761183, 0.759706, 0.526113, 0.508664
                    ],
                    'estimated_true_highstreet_card_pred_prepaid_card': [
                        0.115937, 0.12316, 0.11949, 0.26115, 0.264553
                    ],
                    'estimated_true_highstreet_card_pred_upmarket_card': [
                        0.115792, 0.115657, 0.120804, 0.212737, 0.226783
                    ],
                    'estimated_true_prepaid_card_pred_highstreet_card': [
                        0.124248, 0.124056, 0.122855, 0.150629, 0.143851
                    ],
                    'estimated_true_prepaid_card_pred_prepaid_card': [
                        0.738722, 0.746874, 0.740487, 0.704816, 0.729224
                    ],
                    'estimated_true_prepaid_card_pred_upmarket_card': [
                        0.13703, 0.12907, 0.136658, 0.144555, 0.126925
                    ],
                    'estimated_true_upmarket_card_pred_highstreet_card': [
                        0.104157, 0.099582, 0.099066, 0.175729, 0.161843
                    ],
                    'estimated_true_upmarket_card_pred_prepaid_card': [
                       0.143797, 0.144977, 0.137519, 0.244306, 0.272131
                    ],
                    'estimated_true_upmarket_card_pred_upmarket_card': [
                        0.752046, 0.755441, 0.763415, 0.579965, 0.566026
                    ],
                }
            ),
        ),
    ],
    ids=[
        'size_based_with_timestamp',
    ],
)
def test_iw_for_multiclass_classification_estimation(calculator_opts, expected):
    reference, analysis, _ = load_synthetic_multiclass_classification_dataset()
    cbpe = IW(  # type: ignore
        feature_column_names=[
            "app_behavioral_score",
            "requested_credit_limit",
            "credit_bureau_score",
            "stated_income",
            "acq_channel",
            "app_channel",
            "is_customer"
        ],
        y_true='y_true',
        y_pred='y_pred',
        y_pred_proba={
            'upmarket_card': 'y_pred_proba_upmarket_card',
            'highstreet_card': 'y_pred_proba_highstreet_card',
            'prepaid_card': 'y_pred_proba_prepaid_card',
        },
        problem_type='classification_multiclass',
        metrics=[
            'roc_auc',
            'f1',
            'precision',
            'recall',
            'specificity',
            'accuracy',
            'confusion_matrix',
        ],
        chunk_size=5_000,
        **calculator_opts,
    ).fit(reference.head(15_000))
    result = cbpe.estimate(analysis.tail(10_000))
    column_names = []
    for metric in result.metrics:
        for cname in metric.column_names:
            column_names.append((cname, 'value'))
    sut = result.to_df()[[('chunk', 'key')] + column_names]
    sut.columns = [
        'key',
        'estimated_roc_auc',
        'estimated_f1',
        'estimated_precision',
        'estimated_recall',
        'estimated_specificity',
        'estimated_accuracy',
        'estimated_true_highstreet_card_pred_highstreet_card',
        'estimated_true_highstreet_card_pred_prepaid_card',
        'estimated_true_highstreet_card_pred_upmarket_card',
        'estimated_true_prepaid_card_pred_highstreet_card',
        'estimated_true_prepaid_card_pred_prepaid_card',
        'estimated_true_prepaid_card_pred_upmarket_card',
        'estimated_true_upmarket_card_pred_highstreet_card',
        'estimated_true_upmarket_card_pred_prepaid_card',
        'estimated_true_upmarket_card_pred_upmarket_card',
    ]
    pd.testing.assert_frame_equal(expected, sut.round(6))
