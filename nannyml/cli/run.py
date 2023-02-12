#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import datetime
import warnings
from typing import Any, Dict

import click  # type: ignore
import jinja2  # type: ignore
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from rich.console import Console

from nannyml import runner
from nannyml._typing import ProblemType
from nannyml.chunk import ChunkerFactory, DefaultChunker
from nannyml.cli.cli import cli
from nannyml.config import Config
from nannyml.exceptions import InvalidArgumentsException, IOException
from nannyml.io import DEFAULT_WRITER, FileReader, WriterFactory
from nannyml.io.store import FilesystemStore
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
        # deal with parameter preference: prefer command line above config file
        _ignore_errors = _get_ignore_errors(ignore_errors, config)

        reference_path = _render_path_template(config.input.reference_data.path)
        console.log(f"reading reference data from {reference_path}")
        reference = FileReader(
            filepath=reference_path,
            credentials=config.input.reference_data.credentials,
            read_args=config.input.reference_data.read_args,
        ).read()
        console.log(f"read {reference.size} rows from {reference_path}")

        analysis_path = _render_path_template(config.input.analysis_data.path)
        console.log(f"reading analysis data from {analysis_path}")
        analysis = FileReader(
            filepath=analysis_path,
            credentials=config.input.analysis_data.credentials,
            read_args=config.input.analysis_data.read_args,
        ).read()
        console.log(f"read {analysis.size} rows from {analysis_path}")

        if config.input.target_data:
            target_path = _render_path_template(config.input.target_data.path)
            console.log(f"reading target data from {target_path}")
            targets = FileReader(
                filepath=target_path,
                credentials=config.input.target_data.credentials,
                read_args=config.input.target_data.read_args,
            ).read()
            console.log(f"read {targets.size} rows from {config.input.target_data.path}")
            if config.input.target_data.join_column:
                analysis = analysis.merge(targets, on=config.input.target_data.join_column)
            else:
                analysis = analysis.join(targets)

        if config.output:
            configured_writers = [(w, dict(kwargs)) for w, kwargs in vars(config.output).items() if kwargs is not None]
            if len(configured_writers) > 1:
                raise InvalidArgumentsException(
                    f'only one writer is currently supported ' f'but found {len(configured_writers)}'
                )
            key, kwargs = configured_writers[0]

            if 'path' in kwargs:
                kwargs['path'] = _render_path_template(kwargs['path'])

            console.log(f"using '{key}' writer")
            writer = WriterFactory.create(key, kwargs)
        else:
            console.log(f"no writer config found, falling back to default '{DEFAULT_WRITER.__class__.__name__}' writer")
            writer = DEFAULT_WRITER

        if config.chunker:
            chunker = ChunkerFactory.get_chunker(
                chunk_size=config.chunker.chunk_size,
                chunk_number=config.chunker.chunk_count,
                chunk_period=config.chunker.chunk_period,
                timestamp_column_name=config.column_mapping.dict().get('timestamp', None),
            )
        else:
            chunker = DefaultChunker()
            console.log("no chunker settings specified, using [cyan]default chunker[/]")

        problem_type = ProblemType.parse(config.problem_type)

        store = None
        if config.store:
            if config.store.file:
                store = FilesystemStore(
                    root_path=config.store.file.path,
                    credentials=config.store.file.credentials if 'credentials' in config.store.file else {},
                )

        runner.run(
            reference_data=reference,
            analysis_data=analysis,
            column_mapping=config.column_mapping.dict(),
            problem_type=problem_type,
            chunker=chunker,
            writer=writer,
            run_in_console=True,
            ignore_errors=_ignore_errors,
            store=store,
        )

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


def _get_ignore_errors(ignore_errors: bool, config: Config) -> bool:
    if ignore_errors is None:
        if config.ignore_errors is None:
            return False
        else:
            return config.ignore_errors
    else:
        return ignore_errors


def _render_path_template(path_template: str) -> str:
    try:
        env = jinja2.Environment()
        tpl = env.from_string(path_template)
        return tpl.render(
            minute=datetime.datetime.strftime(datetime.datetime.today(), "%M"),
            hour=datetime.datetime.strftime(datetime.datetime.today(), "%H"),
            day=datetime.datetime.strftime(datetime.datetime.today(), "%d"),
            weeknumber=datetime.date.today().isocalendar()[1],
            month=datetime.datetime.strftime(datetime.datetime.today(), "%m"),
            year=datetime.datetime.strftime(datetime.datetime.today(), "%Y"),
        )
    except Exception as exc:
        raise IOException(f"could not render file path template: '{path_template}': {exc}")


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
