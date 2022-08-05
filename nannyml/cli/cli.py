#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

import click  # type: ignore
from rich.console import Console

from nannyml import __version__
from nannyml.config import Config


@click.group()
@click.pass_context
@click.version_option(__version__, '--version', '-v')
@click.option(
    '-c',
    '--configuration-path',
    type=click.Path(),
    help='Path to your NannyML configuration file',
)
def cli(ctx, configuration_path) -> None:
    """CLI root command."""

    # setting up click.context
    ctx.obj = {}

    # loading configuration
    config = Config.load(configuration_path)
    ctx.obj['config'] = config

    # setting up Rich console
    console = Console()
    ctx.obj['console'] = console


if __name__ == "__main__":
    cli()
