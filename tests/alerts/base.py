#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Tests for the nannyml.alerts.base module."""
import re

import pytest

from nannyml import SlackNotificationHandler
from nannyml.alerts import AlertHandler, AlertHandlerFactory
from nannyml.exceptions import InvalidArgumentsException


@pytest.fixture()
def factory() -> AlertHandlerFactory:
    factory = AlertHandlerFactory()
    if 'foo' in factory.registry:
        del factory.registry['foo']
    return factory


def test_alert_handler_factory_register_shows_warning_when_using_existing_key(factory):
    key = 'foo'
    factory.register(key)(AlertHandler)
    with pytest.warns(UserWarning, match=f'an AlertHandler was already registered for key {key} and will be replaced.'):
        AlertHandlerFactory.register(key)(AlertHandler)


def test_alert_handler_factory_create_raises_exception_when_given_non_existent_key(factory):
    key = 'foo'
    with pytest.raises(
        InvalidArgumentsException, match=re.escape(f"unknown metric key '{key}' given. Should be one of ['slack'].")
    ):
        _ = factory.create(key)


def test_alert_handler_factory_creates_non_existing_error_options_include_custom_new_entries(factory):
    factory.register('foo')(AlertHandler)
    with pytest.raises(
        InvalidArgumentsException,
        match=re.escape("unknown metric key 'bar' given. Should be one of ['slack', 'foo']."),
    ):
        _ = factory.create(key='bar')


@pytest.mark.parametrize('key, kwargs, expected', [('slack', dict(webhook_url='url'), SlackNotificationHandler)])
def test_alert_handler_factory_create_returns_correct_alert_handler_when_given_correct_key(
    factory, key, kwargs, expected
):
    sut = factory.create(key, **kwargs)
    assert type(sut) == expected
