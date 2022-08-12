#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

import click  # type: ignore
from rich.console import Console

from nannyml import runner
from nannyml.cli.cli import cli
from nannyml.io.file_reader import FileReader
from nannyml.io.file_writer import FileWriter


@cli.command()
@click.pass_context
@click.option(
    '--ignore-errors',
    is_flag=True,
    flag_value=True,
    default=None,
    help='Continues with the next calculation if the previous one errors out',
)
def run(ctx, ignore_errors: bool):
    config = ctx.obj['config']

    console = Console()

    # deal with parameter preference: prefer command line above config file
    if ignore_errors is None:
        if config.ignore_errors is None:
            ignore_errors = False
        else:
            ignore_errors = config.ignore_errors

    console.log(f"reading reference data from {config.input.reference_data.path}")
    reference = FileReader(
        filepath=config.input.reference_data.path,
        credentials=config.input.reference_data.credentials,
        read_args=config.input.reference_data.read_args,
    ).read()
    console.log(f"read {reference.size} rows from {config.input.reference_data.path}")

    console.log(f"reading analysis data from {config.input.analysis_data.path}")
    analysis = FileReader(
        filepath=config.input.analysis_data.path,
        credentials=config.input.analysis_data.credentials,
        read_args=config.input.analysis_data.read_args,
    ).read()
    console.log(f"read {analysis.size} rows from {config.input.analysis_data.path}")

    if config.input.target_data:
        console.log(f"reading target data from {config.input.target_data.path}")
        targets = FileReader(
            filepath=config.input.target_data.path,
            credentials=config.input.target_data.credentials,
            read_args=config.input.target_data.read_args,
        ).read()
        console.log(f"read {targets.size} rows from {config.input.target_data.path}")
        if config.input.target_data.join_column:
            analysis = analysis.merge(targets, on=config.input.target_data.join_column)
        else:
            analysis = analysis.merge(targets)

    writer = FileWriter(
        filepath=config.output.path,
        data_format=config.output.format,
        credentials=config.output.credentials,
        write_args=config.output.write_args,
    )

    runner.run(
        reference_data=reference,
        analysis_data=analysis,
        column_mapping=config.column_mapping.dict(),
        writer=writer,
        run_in_console=True,
        ignore_errors=ignore_errors,
    )
