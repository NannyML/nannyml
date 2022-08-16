#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import datetime

import click  # type: ignore
import jinja2  # type: ignore
from rich.console import Console

from nannyml import runner
from nannyml.chunk import ChunkerFactory, DefaultChunker
from nannyml.cli.cli import cli
from nannyml.config import Config
from nannyml.exceptions import IOException
from nannyml.io.file_reader import FileReader
from nannyml.io.file_writer import FileWriter


@cli.command()
@click.pass_context
@click.option(
    '--ignore-errors',
    is_flag=True,
    flag_value=True,
    default=None,
    help='Continues the next calculation if the previous one errors out',
)
def run(ctx, ignore_errors: bool):
    config = ctx.obj['config']

    console = Console()

    # deal with parameter preference: prefer command line above config file
    _ignore_errors = _get_ignore_errors(ignore_errors, config)

    reference_path = _render_path_template(config.input.reference_data.path)
    console.log(f"reading reference data from {reference_path}")
    reference = FileReader(
        filepath=reference_path,
        credentials=config.input.reference_data.credentials,
        read_args=config.input.reference_data.read_args,
    ).read()
    console.log(f"read {reference.size} rows from {config.input.reference_data.path}")

    analysis_path = _render_path_template(config.input.analysis_data.path)
    console.log(f"reading analysis data from {analysis_path}")
    analysis = FileReader(
        filepath=analysis_path,
        credentials=config.input.analysis_data.credentials,
        read_args=config.input.analysis_data.read_args,
    ).read()
    console.log(f"read {analysis.size} rows from {config.input.analysis_data.path}")

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
            analysis = analysis.merge(targets)

    writer = FileWriter(
        filepath=_render_path_template(config.output.path),
        data_format=config.output.format,
        credentials=config.output.credentials,
        write_args=config.output.write_args,
    )

    if config.chunker:
        chunker = ChunkerFactory.get_chunker(
            chunk_size=config.chunker.chunk_size,
            chunk_number=config.chunker.chunk_count,
            chunk_period=config.chunker.chunk_period,
        )
    else:
        chunker = DefaultChunker()
        console.log("no chunker settings specified, using [cyan]default chunker[/]")

    runner.run(
        reference_data=reference,
        analysis_data=analysis,
        column_mapping=config.column_mapping.dict(),
        chunker=chunker,
        writer=writer,
        run_in_console=True,
        ignore_errors=_ignore_errors,
    )


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
            minute=datetime.datetime.today().minute,
            hour=datetime.datetime.today().hour,
            day=datetime.datetime.today().day,
            weeknumber=datetime.date.today().isocalendar()[1],
            month=datetime.date.today().month,
            year=datetime.date.today().year,
        )
    except Exception as exc:
        raise IOException(f"could not render file path template: '{path_template}': {exc}")
