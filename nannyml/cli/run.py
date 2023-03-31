#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import datetime
import warnings
from typing import Any, Dict

import click
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from rich.console import Console

from nannyml import runner
from nannyml.cli.cli import cli
from nannyml.config import Config
from nannyml.exceptions import InvalidArgumentsException
from nannyml.usage_logging import UsageEvent, log_usage


@cli.command()
@click.pass_context
@click.option(
    '--ignore-errors',
    is_flag=True,
    flag_value=True,
    default=None,
    help='Continues the next calculation if the previous one errors out',
)
@log_usage(UsageEvent.CLI_RUN)
def run(ctx, ignore_errors: bool):
    config = ctx.obj['config']

    console = Console()

    def actually_run_it():
        runner.run(config=config, console=console)

        if config.scheduling:
            next_run_time = scheduler.get_job(job_id='nml_run').next_run_time
            console.log(f"run successfully completed, sleeping until next run at {next_run_time}")

    if not config.scheduling:
        console.log("no scheduler configured, performing one-off run")
        actually_run_it()
    else:
        with warnings.catch_warnings():  # filter out some deprecation warnings in APscheduler
            warnings.simplefilter("ignore")
            scheduler = BlockingScheduler()
            trigger_args = _build_scheduling_trigger_args(config)
            scheduler.add_job(actually_run_it, id='nml_run', **trigger_args)

            try:
                console.log(f"starting scheduler with trigger args {trigger_args}")
                scheduler.start()
            except KeyboardInterrupt:
                pass
            except SystemExit:
                pass


def _build_scheduling_trigger_args(config: Config) -> Dict[str, Any]:
    if not config.scheduling:
        return {}

    if not config.scheduling.cron and not config.scheduling.interval:
        raise InvalidArgumentsException(
            "found no subsections in the scheduling configuration. "
            "To disable scheduling also remove the 'scheduling' section."
        )

    if config.scheduling.cron and config.scheduling.interval:
        raise InvalidArgumentsException(
            "found multiple subsections in the scheduling configuration." "Only one should be present."
        )

    if config.scheduling.cron:
        return {'trigger': CronTrigger.from_crontab(config.scheduling.cron.crontab)}
    elif config.scheduling.interval:
        if not config.scheduling.interval.dict():
            raise InvalidArgumentsException(
                "found no values for the 'scheduling.interval' section. " "Provide at least one interval value."
            )
        not_none_intervals = {k: v for k, v in config.scheduling.interval.dict().items() if v is not None}
        if len(not_none_intervals) > 1:
            raise InvalidArgumentsException(
                "found multiple values in the 'scheduling.interval' section. " "Provide exactly one interval value."
            )
        return {'trigger': 'interval', **not_none_intervals, 'next_run_time': datetime.datetime.now()}

    return {}
