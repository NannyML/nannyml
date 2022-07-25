#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from nannyml.cli.cli import cli


@cli.command()
def run():
    print('running NML')
