#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import copy

import pytest
import slack_sdk.webhook
from mock import mock
from pytest_lazyfixture import lazy_fixture

from nannyml.alerts import BlocksBuilder, SlackNotificationHandler
from nannyml.datasets import load_synthetic_binary_classification_dataset, load_synthetic_car_price_dataset
from nannyml.drift.multivariate.data_reconstruction import DataReconstructionDriftCalculator
from nannyml.drift.multivariate.data_reconstruction import Result as MultivariateDriftResult
from nannyml.drift.univariate import Result as UnivariateDriftResult
from nannyml.drift.univariate import UnivariateDriftCalculator
from nannyml.exceptions import AlertHandlerException, InvalidArgumentsException
from nannyml.performance_calculation import PerformanceCalculator
from nannyml.performance_calculation import Result as RealizedPerformanceResult
from nannyml.performance_estimation.confidence_based import CBPE
from nannyml.performance_estimation.confidence_based import Result as CBPEResult
from nannyml.performance_estimation.direct_loss_estimation import DLE
from nannyml.performance_estimation.direct_loss_estimation import Result as DLEResult


@pytest.fixture(scope='module')
def univariate_result() -> UnivariateDriftResult:
    reference, analysis, _ = load_synthetic_binary_classification_dataset()
    calc = UnivariateDriftCalculator(
        column_names=[
            col for col in reference.columns if col not in ['timestamp', 'identifier', 'work_home_actual', 'period']
        ],
        timestamp_column_name='timestamp',
        continuous_methods=['wasserstein', 'jensen_shannon'],
        categorical_methods=['l_infinity', 'jensen_shannon'],
    ).fit(reference)
    return calc.calculate(analysis)


@pytest.fixture(scope='module')
def multivariate_drift_result() -> MultivariateDriftResult:
    reference, analysis, _ = load_synthetic_binary_classification_dataset()
    calc = DataReconstructionDriftCalculator(
        column_names=[
            col for col in reference.columns if col not in ['timestamp', 'identifier', 'work_home_actual', 'period']
        ],
        timestamp_column_name='timestamp',
    ).fit(reference)
    return calc.calculate(analysis)


@pytest.fixture(scope='module')
def realized_performance_result() -> RealizedPerformanceResult:
    reference, analysis, analysis_targets = load_synthetic_binary_classification_dataset()
    calc = PerformanceCalculator(
        metrics=['roc_auc', 'f1'],
        y_true='work_home_actual',
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        timestamp_column_name='timestamp',
        problem_type='classification',
    ).fit(reference)
    return calc.calculate(analysis.merge(analysis_targets, on='identifier'))


@pytest.fixture(scope='module')
def cbpe_result() -> CBPEResult:
    reference, analysis, _ = load_synthetic_binary_classification_dataset()
    est = CBPE(  # type: ignore
        metrics=['roc_auc', 'f1'],
        y_true='work_home_actual',
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        timestamp_column_name='timestamp',
        problem_type='classification',
    ).fit(reference)
    return est.estimate(analysis)


@pytest.fixture(scope='module')
def dle_result() -> DLEResult:
    reference, analysis, _ = load_synthetic_car_price_dataset()
    est = DLE(
        feature_column_names=[col for col in reference.columns if col not in ['y_pred', 'y_true', 'timestamp']],
        y_pred='y_pred',
        y_true='y_true',
        metrics=['mae', 'mse'],
        timestamp_column_name='timestamp',
    ).fit(reference)
    return est.estimate(analysis)


def test_handle_calls_slack_send(univariate_result):
    sut = SlackNotificationHandler(webhook_url='https://hooks.slack.com/services/some/random/identifier')

    with mock.patch.object(sut, '_client') as patched_client:
        patched_client.send.return_value = slack_sdk.webhook.WebhookResponse(
            status_code=200, headers={}, body='', url=''
        )

        sut.handle(results=univariate_result)

        patched_client.send.assert_called_once()


def test_handle_raises_exception_when_slack_send_doesnt_return_status_200(univariate_result):
    sut = SlackNotificationHandler(webhook_url='https://hooks.slack.com/services/some/random/identifier')

    with mock.patch.object(sut, '_client') as patched_client:
        patched_client.send.return_value = slack_sdk.webhook.WebhookResponse(
            status_code=404, headers={}, body='foo', url=''
        )

        with pytest.raises(AlertHandlerException, match="Slack returned status code '404': 'foo'"):
            sut.handle(results=univariate_result)


def test_handle_raises_invalid_arguments_exception_when_given_non_existing_webhook_url(univariate_result):
    sut = SlackNotificationHandler(webhook_url='https://hooks.slack.com/services/some/random/identifier')
    with pytest.raises(InvalidArgumentsException, match="invalid webhook_url"):
        sut.handle(results=univariate_result)


def test_handle_raises_handler_exception_when_given_invalid_webhook_url(univariate_result):
    sut = SlackNotificationHandler(webhook_url='not_a_real_url')
    with pytest.raises(AlertHandlerException, match="an unexpected exception occurred upon calling Slack"):
        sut.handle(results=univariate_result)


def test_handle_raises_handler_exception_when_slack_call_errors_out(univariate_result):
    sut = SlackNotificationHandler(webhook_url='https://hooks.slack.com/services/some/random/identifier')
    with mock.patch.object(sut, '_client') as patched_client:
        patched_client.send.side_effect = RuntimeError()
        with pytest.raises(AlertHandlerException, match="an unexpected exception occurred upon calling Slack"):
            sut.handle(results=univariate_result)


def test_block_builder_sets_checkmark_icon_when_no_alerts_in_univariate_drift_result(univariate_result):
    result = copy.deepcopy(univariate_result)
    result.to_df().loc[:, (slice(None), slice(None), 'alert')] = False
    result_summary_block = BlocksBuilder().add_result(result, only_alerts=False).build()[0]
    sut = result_summary_block['text']['text']
    assert isinstance(sut, str)
    assert sut.startswith(':white_check_mark:')


def test_block_builder_doesnt_add_details_block_when_no_alerts_in_univariate_drift_result(univariate_result):
    result = copy.deepcopy(univariate_result)
    result.to_df().loc[:, (slice(None), slice(None), 'alert')] = False
    blocks = BlocksBuilder().add_result(result, only_alerts=False).build()
    assert len(blocks) == 1


def test_block_builder_sets_warning_icon_when_alerts_in_univariate_drift_result(univariate_result):
    result_summary_block = BlocksBuilder().add_result(univariate_result, only_alerts=False).build()[0]
    sut = result_summary_block['text']['text']
    assert isinstance(sut, str)
    assert sut.startswith(':warning:')


def test_block_builder_adds_drifting_columns_in_details_block_when_alerts_in_univariate_drift_result(univariate_result):
    blocks = BlocksBuilder().add_result(univariate_result, only_alerts=False).build()
    assert len(blocks) == 2
    sut = blocks[1]['text']['text']
    assert isinstance(sut, str)
    for col in [
        'y_pred_proba',
        'y_pred',
        'wfh_prev_workday',
        'tenure',
        'salary_range',
        'public_transportation_cost',
        'distance_from_office',
        'gas_price_per_litre',
    ]:
        assert col in sut


@pytest.mark.parametrize(
    'result',
    [
        lazy_fixture('multivariate_drift_result'),
        lazy_fixture('realized_performance_result'),
        lazy_fixture('cbpe_result'),
        lazy_fixture('dle_result'),
    ],
)
def test_block_builder_sets_checkmark_icon_when_no_alerts_in_result(result):
    result = copy.deepcopy(result)
    result.to_df().loc[:, (slice(None), 'alert')] = False
    result_summary_block = BlocksBuilder().add_result(result, only_alerts=False).build()[0]
    sut = result_summary_block['text']['text']
    assert isinstance(sut, str)
    assert sut.startswith(':white_check_mark:')


@pytest.mark.parametrize(
    'result',
    [
        lazy_fixture('multivariate_drift_result'),
        lazy_fixture('realized_performance_result'),
        lazy_fixture('cbpe_result'),
        lazy_fixture('dle_result'),
    ],
)
def test_block_builder_doesnt_add_details_block_when_no_alerts_in_result(result):
    result = copy.deepcopy(result)
    result.to_df().loc[:, (slice(None), 'alert')] = False
    blocks = BlocksBuilder().add_result(result, only_alerts=False).build()
    assert len(blocks) == 1


@pytest.mark.parametrize(
    'result',
    [
        lazy_fixture('multivariate_drift_result'),
        lazy_fixture('realized_performance_result'),
        lazy_fixture('cbpe_result'),
        lazy_fixture('dle_result'),
    ],
)
def test_block_builder_sets_warning_icon_when_alerts_in_result(result):
    result_summary_block = BlocksBuilder().add_result(result, only_alerts=False).build()[0]
    sut = result_summary_block['text']['text']
    assert isinstance(sut, str)
    assert sut.startswith(':warning:')


@pytest.mark.parametrize(
    'result',
    [
        lazy_fixture('multivariate_drift_result'),
        lazy_fixture('realized_performance_result'),
        lazy_fixture('cbpe_result'),
        lazy_fixture('dle_result'),
    ],
)
def test_block_builder_adds_alerting_metrics_in_details_block_when_alerts_in_result(result):
    blocks = BlocksBuilder().add_result(result, only_alerts=False).build()
    assert len(blocks) == 2
    sut = blocks[1]['text']['text']
    assert isinstance(sut, str)
    for metric in result.metrics:
        assert metric.display_name in sut
