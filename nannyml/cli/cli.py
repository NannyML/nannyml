#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import os

import click  # type: ignore
from art import text2art
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
@click.option(
    '--disable-usage-analytics', is_flag=True, flag_value=True, default=False, help='Disable collecting usage analytics'
)
def cli(ctx, configuration_path, disable_usage_analytics: bool) -> None:
    """CLI root command."""

    # setting up click.context
    ctx.obj = {}

    # setting up Rich console
    console = Console()
    console.print(
        f"[cyan]{text2art('NannyML', font='doom')}[/]",
    )

    # loading configuration
    console.log(f"loading configuration file from {get_config_path(configuration_path).absolute()}")
    config = Config.load(configuration_path)
    ctx.obj['config'] = config

    # disable usage analytics if need be
    if disable_usage_analytics is None:
        if config.disable_usage_analytics is None:
            disable_usage_analytics = False
        else:
            disable_usage_analytics = config.disable_usage_analytics
    if disable_usage_analytics:
        console.log('disabling usage analytics')
        os.environ['NML_DISABLE_USAGE_ANALYTICS'] = '1'


if __name__ == "__main__":
    cli()
