#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import datetime
import logging
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Union

import jinja2

from nannyml._typing import Result
from nannyml.alerts import AlertHandler, AlertHandlerFactory, get_column_names_with_alerts, get_metrics_with_alerts
from nannyml.drift.univariate import Result as UnivariateDriftResult
from nannyml.exceptions import AlertHandlerException


@AlertHandlerFactory.register(key='email')
class EmailNotificationHandler(AlertHandler):
    def __init__(
        self,
        smtp_server_address: str,
        smtp_server_port: int,
        smtp_from_address: str,
        smtp_password: str,
        smtp_to_addresses: List[str],
    ):
        self.smtp_server_address = smtp_server_address
        self.smtp_server_port = smtp_server_port
        self.smtp_from_address = smtp_from_address
        self.smtp_password = smtp_password
        self.smtp_to_addresses = smtp_to_addresses

        try:
            template_loader = jinja2.PackageLoader('nannyml')
            self.template_environment = jinja2.Environment(
                loader=template_loader, autoescape=True, trim_blocks=True, lstrip_blocks=True
            )
        except Exception as exc:
            raise AlertHandlerException(f'could not load email templates: {exc}')

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def handle(self, results: Union[Result, List[Result]], only_alerts: bool = True):
        if not isinstance(results, List):
            results = [results]

        try:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(self.smtp_server_address, self.smtp_server_port, context=context) as server:
                server.login(self.smtp_from_address, self.smtp_password)
                try:
                    self._logger.debug('sending email notification')
                    server.sendmail(
                        from_addr=self.smtp_from_address, to_addrs=self.smtp_to_addresses, msg=self.get_message(results)
                    )
                finally:
                    self._logger.debug('quitting SMTP connection')
                    server.quit()
        except Exception as exc:
            self._logger.warning(f'failed to send email notification: {exc}')
            # raise exc

    def get_message(self, results: List[Result]) -> str:
        message = MIMEMultipart("alternative")
        message["Subject"] = f'NannyML daily summary for {datetime.date.today().isoformat()}'

        message_data = _result_list_to_params(results)

        plain_text_template = self.template_environment.get_template('email/daily_summary_plain.jinja')
        plain_text = plain_text_template.render(message_data)
        message.attach(MIMEText(plain_text, "plain"))

        html_template = self.template_environment.get_template('email/daily_summary_html.jinja')
        html = html_template.render(message_data)
        message.attach(MIMEText(html, "html"))

        return message.as_string()


def _result_list_to_params(results: List[Result]) -> Dict[str, Any]:
    return {
        'date_of_today': datetime.date.today().isoformat(),
        'alerts': {f'{result.__module__}.{result.__class__.__name__}': _result_to_params(result) for result in results},
    }


def _result_to_params(result: Result) -> Dict[str, Any]:
    res: Dict[str, Any] = {}

    if isinstance(result, UnivariateDriftResult):
        res['alerting_items'] = get_column_names_with_alerts(result)
    else:
        res['alerting_items'] = get_metrics_with_alerts(result)

    res['alert_count'] = len(res['alerting_items'])
    res['has_alerts'] = len(res['alerting_items']) > 0

    return res
