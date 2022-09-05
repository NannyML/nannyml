#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  #
#  License: Apache Software License 2.0

#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import json
from typing import List

from docutils.nodes import Node, paragraph  # type: ignore
from docutils.parsers.rst.directives import path, positive_int  # type: ignore
from docutils.statemachine import StringList  # type: ignore
from sphinx.util.docutils import SphinxDirective


class NbTable(SphinxDirective):

    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {
        'path': path,
        'cell': positive_int,
    }
    has_content = False

    def run(self) -> List[Node]:
        with open(self.options['path']) as nb_file:
            nb = json.load(nb_file)
            output = nb['cells'][self.options['cell']]['outputs'][0]['text']
            output_lines = output

        content = f".. table::\n\n{''.join(['    ' + line for line in output_lines])}\n\n"

        view = StringList(content.split('\n'))
        node = paragraph(rawsource=content)
        self.state.nested_parse(view, self.content_offset, node)
        return [node]


def setup(app):
    app.add_directive("nbtable", NbTable)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
