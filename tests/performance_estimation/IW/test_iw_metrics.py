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
                    'estimated_roc_auc': [0.972678, 0.972252, 0.972055, 0.972685, 0.971584],
                    'estimated_f1': [0.942237, 0.941807, 0.943934, 0.937973, 0.938716],
                    'estimated_precision': [0.957635, 0.956823, 0.957734, 0.950811, 0.948012],
                    'estimated_recall': [0.927327, 0.927255, 0.930525, 0.925478, 0.9296],
                    'estimated_specificity': [0.959997, 0.959452, 0.958218, 0.952975, 0.94577],
                    'estimated_accuracy': [0.943868, 0.943606, 0.944251, 0.93935, 0.937436],
                    'estimated_true_positive': [6679.18566, 6649.376347, 6837.673942, 4954.564752, 5164.873846],
                    'estimated_true_negative': [7091.125729, 7099.804624, 6920.243971, 5194.346387, 4939.721549],
                    'estimated_false_positive': [295.484659, 300.0523, 301.752944, 256.318617, 283.239176],
                    'estimated_false_negative': [523.433173, 521.659873, 510.514901, 398.956322, 391.141989],
                    'estimated_business_value': [74262.353029, 73976.512484, 75603.313631, 54663.183996, 56200.565786],
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
                    'estimated_roc_auc': [0.905758, 0.905864, 0.906262, 0.796482, 0.791788],
                    'estimated_f1': [0.752541, 0.752992, 0.753604, 0.60351, 0.602515],
                    'estimated_precision': [0.752567, 0.753051, 0.753657, 0.606623, 0.607366],
                    'estimated_recall': [0.752519, 0.752945, 0.753561, 0.603631, 0.601305],
                    'estimated_specificity': [0.876199, 0.876384, 0.87676, 0.803273, 0.803623],
                    'estimated_accuracy': [0.752466, 0.752871, 0.753616, 0.610914, 0.614981],
                    'estimated_true_highstreet_card_pred_highstreet_card': [
                        0.762972, 0.760708, 0.760166, 0.526113, 0.508664
                    ],
                    'estimated_true_highstreet_card_pred_prepaid_card': [
                        0.118261, 0.122407, 0.119831, 0.26115, 0.264553
                    ],
                    'estimated_true_highstreet_card_pred_upmarket_card': [
                        0.118767, 0.116885, 0.120003, 0.212737, 0.226783
                    ],
                    'estimated_true_prepaid_card_pred_highstreet_card': [
                        0.124961, 0.124416, 0.125601, 0.150629, 0.143851
                    ],
                    'estimated_true_prepaid_card_pred_prepaid_card': [
                        0.739897, 0.745947, 0.73924, 0.704816, 0.729224
                    ],
                    'estimated_true_prepaid_card_pred_upmarket_card': [
                        0.135142, 0.129637, 0.135159, 0.144555, 0.126925
                    ],
                    'estimated_true_upmarket_card_pred_highstreet_card': [
                        0.102871, 0.102934, 0.101066, 0.175729, 0.161843
                    ],
                    'estimated_true_upmarket_card_pred_prepaid_card': [
                       0.142441, 0.144888, 0.137655, 0.244306, 0.272131
                    ],
                    'estimated_true_upmarket_card_pred_upmarket_card': [
                        0.754688, 0.752178, 0.761279, 0.579965, 0.566026
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
