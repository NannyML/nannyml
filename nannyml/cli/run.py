#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

import click  # type: ignore

from nannyml import runner
from nannyml.cli.cli import cli
from nannyml.io.file_reader import FileReader
from nannyml.io.file_writer import FileWriter


@cli.command()
@click.pass_context
def run(ctx):
    config = ctx.obj['config']
    reference = FileReader(
        filepath=config.input.reference_data.path,
        credentials=config.input.reference_data.credentials,
        read_args=config.input.reference_data.read_args,
    ).read()

    analysis = FileReader(
        filepath=config.input.analysis_data.path,
        credentials=config.input.analysis_data.credentials,
        read_args=config.input.analysis_data.read_args,
    ).read()

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
    )
