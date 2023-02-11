#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

import click
from pyfiglet import Figlet
from rich.console import Console

from nannyml import __version__
from nannyml.config import Config, get_config_path


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

    # setting up Rich console
    console = Console()
    console.print(
        f"[cyan]{Figlet(font='slant').renderText('NannyML')}[/]",
    )

    # loading configuration
    console.log(f"loading configuration file from {get_config_path(configuration_path).absolute()}")
    config = Config.load(configuration_path)
    ctx.obj['config'] = config


if __name__ == "__main__":
    cli()
