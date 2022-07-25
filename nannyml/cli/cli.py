#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

import click  # type: ignore

from nannyml import __version__


@click.group()
@click.version_option(__version__, '--version', '-v')
def cli() -> None:
    """CLI root command."""


if __name__ == "__main__":
    cli()
