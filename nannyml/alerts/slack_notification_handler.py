#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Union

from slack_sdk import WebhookClient

from nannyml._typing import Result
from nannyml.drift.multivariate.data_reconstruction.result import Result as DataReconstructionDriftResult
from nannyml.drift.univariate import Result as UnivariateDriftResult
from nannyml.performance_calculation.result import Result as RealizedPerformanceResult
from nannyml.performance_estimation.confidence_based.results import Result as CBPEResult
from nannyml.performance_estimation.direct_loss_estimation.result import Result as DLEResult


class SlackNotificationHandler:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self._client = WebhookClient(url=webhook_url)
        self._builder = BlocksBuilder()

    def handle(self, results: Union[Result, List[Result]], only_alerts: bool = True):
        if not isinstance(results, List):
            results = [results]

        self._builder.add_header().add_divider()

        for result in results:
            self._builder.add_result(result, only_alerts).add_divider()

        self._client.send(blocks=self._builder.build())


class BlocksBuilder:
    def __init__(self):
        self._blocks: List[Dict[str, Any]] = []

    def add_header(self) -> BlocksBuilder:
        self._blocks += [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"Daily summary for *<!date^{int(datetime.timestamp(datetime.now()))}^{{date}}|today>*",
                },
            },
        ]
        return self

    def add_divider(self) -> BlocksBuilder:
        self._blocks += [{"type": "divider"}]
        return self

    def add_result(self, result: Result, only_alerts: bool) -> BlocksBuilder:
        if isinstance(result, UnivariateDriftResult):
            self._blocks += _univariate_drift_result_blocks(result, only_alerts)
        elif isinstance(result, DataReconstructionDriftResult):
            self._blocks += _multivariate_drift_result_blocks(result, only_alerts)
        elif isinstance(result, RealizedPerformanceResult):
            self._blocks += _realized_performance_result_blocks(result, only_alerts)
        elif isinstance(result, CBPEResult):
            self._blocks += _estimated_performance_cbpe_result_blocks(result, only_alerts)
        elif isinstance(result, DLEResult):
            self._blocks += _estimated_performance_dle_result_blocks(result, only_alerts)
        return self

    def build(self) -> List[Dict[str, Any]]:
        return self._blocks


def _univariate_drift_result_blocks(result: UnivariateDriftResult, only_alerts: bool) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []

    df = result.filter(period='analysis').to_df()

    columns_with_alerts = set()
    for column_name in result.column_names:
        alert_cols = df.loc[:, (column_name, slice(None), 'alert')].columns
        for col in alert_cols:
            has_alerts = df.get(col).any()
            if has_alerts:
                columns_with_alerts.add(column_name)

    icon = ':white_check_mark:' if len(columns_with_alerts) == 0 else ':warning:'
    blocks.append(
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"{icon} *Univariate Drift* "
                + (f"- {len(columns_with_alerts)} columns drifting" if len(columns_with_alerts) > 0 else ''),
            },
        }
    )
    drifting_columns_str = '\n'.join([f'• {col}' for col in columns_with_alerts])
    if len(columns_with_alerts) > 0:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": drifting_columns_str}})
    return blocks


def _multivariate_drift_result_blocks(result: DataReconstructionDriftResult, only_alerts: bool) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    df = result.filter(period='analysis').to_df()

    has_alerts = df.get(('reconstruction_error', 'alert')).sum() > 0
    icon = ':white_check_mark:' if not has_alerts else ':warning:'
    blocks.append(
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"{icon} *Multivariate Drift* " + "- 1 metric unacceptable" if has_alerts else '',
            },
        }
    )
    if has_alerts:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": '• Reconstruction error'}})
    return blocks


def _realized_performance_result_blocks(result: RealizedPerformanceResult, only_alerts: bool) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    df = result.filter(period='analysis').to_df()

    metrics_with_alerts = set()
    for metric in result.metrics:
        has_alerts = df.get((metric.column_name, 'alert')).any()
        if has_alerts:
            metrics_with_alerts.add(metric.display_name)
    icon = ':white_check_mark:' if len(metrics_with_alerts) == 0 else ':warning:'
    blocks.append(
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"{icon} *Realized Performance* "
                + (f"- {len(metrics_with_alerts)} metrics unacceptable" if len(metrics_with_alerts) > 0 else ''),
            },
        }
    )
    drifting_metrics_str = '\n'.join([f'• {m}' for m in metrics_with_alerts])
    if len(metrics_with_alerts) > 0:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": drifting_metrics_str}})
    return blocks


def _estimated_performance_cbpe_result_blocks(result: CBPEResult, only_alerts: bool) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    df = result.filter(period='analysis').to_df()

    metrics_with_alerts = set()
    for metric in result.metrics:
        has_alerts = df.get((metric.column_name, 'alert')).any()
        if has_alerts:
            metrics_with_alerts.add(metric.display_name)
    icon = ':white_check_mark:' if len(metrics_with_alerts) == 0 else ':warning:'
    blocks.append(
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"{icon} *Estimated Performance (CBPE)* "
                + (f"- {len(metrics_with_alerts)} metrics unacceptable" if len(metrics_with_alerts) > 0 else ''),
            },
        }
    )
    drifting_metrics_str = '\n'.join([f'• {m}' for m in metrics_with_alerts])
    if len(metrics_with_alerts) > 0:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": drifting_metrics_str}})
    return blocks


def _estimated_performance_dle_result_blocks(result: DLEResult, only_alerts: bool) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    df = result.filter(period='analysis').to_df()

    metrics_with_alerts = set()
    for metric in result.metrics:
        has_alerts = df.get((metric.column_name, 'alert')).any()
        if has_alerts:
            metrics_with_alerts.add(metric.display_name)
    icon = ':white_check_mark:' if len(metrics_with_alerts) == 0 else ':warning:'
    blocks.append(
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"{icon} *Estimated Performance (DLE)* "
                + (f"- {len(metrics_with_alerts)} metrics unacceptable" if len(metrics_with_alerts) > 0 else ''),
            },
        }
    )
    drifting_metrics_str = '\n'.join([f'• {m}' for m in metrics_with_alerts])
    if len(metrics_with_alerts) > 0:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": drifting_metrics_str}})
    return blocks
