#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

import mock
import pytest
import slack.webhook

from nannyml.alerts import SlackNotificationHandler
from nannyml.datasets import load_synthetic_binary_classification_dataset
from nannyml.drift.univariate import Result as UnivariateDriftResult
from nannyml.drift.univariate import UnivariateDriftCalculator
from nannyml.exceptions import AlertHandlerException, InvalidArgumentsException


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


def test_handle_calls_slack_send(univariate_result):
    sut = SlackNotificationHandler(webhook_url='https://hooks.slack.com/services/some/random/identifier')

    with mock.patch.object(sut, '_client') as patched_client:
        patched_client.send.return_value = slack.webhook.WebhookResponse(status_code=200, headers={}, body='', url='')

        sut.handle(results=univariate_result)

        patched_client.send.assert_called_once()


def test_handle_raises_exception_when_slack_send_doesnt_return_status_200(univariate_result):
    sut = SlackNotificationHandler(webhook_url='https://hooks.slack.com/services/some/random/identifier')

    with mock.patch.object(sut, '_client') as patched_client:
        patched_client.send.return_value = slack.webhook.WebhookResponse(
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
