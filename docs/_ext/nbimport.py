#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  #
#  License: Apache Software License 2.0

#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import json
from typing import List

from docutils import nodes  # type: ignore
from docutils.parsers.rst.directives import flag, path, positive_int_list  # type: ignore
from sphinx.directives.code import CodeBlock
from sphinx.util.docutils import SphinxDirective


class NbImport(SphinxDirective):

    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {
        'path': path,
        'cells': positive_int_list,
        'hide_source': flag,
        'hide_prompts': flag,
        'show_output': flag,
    }
    has_content = False

    def run(self) -> List[nodes.Node]:
        cell_content: List[str] = []
        with open(self.options['path']) as nb_file:
            try:
                nb = json.load(nb_file)
                for cell_index in self.options['cells']:
                    if 'hide_source' not in self.options:
                        source_lines = nb['cells'][cell_index - 1]['source']
                        source_lines = [line.replace('\n', '') for line in source_lines]
                        if 'hide_prompts' not in self.options:
                            cell_content += [_add_prompts(line) for line in source_lines]
                        else:
                            cell_content += source_lines
                    if 'show_output' in self.options:
                        outputs = nb['cells'][cell_index - 1]['outputs'][0]
                        if 'text' in outputs:
                            output = outputs['text']
                        elif 'data' in outputs:
                            output = outputs['data']['text/plain']
                        if isinstance(output, str):
                            cell_content += [output]
                        else:
                            cell_content += [line.replace('\n', '') for line in output]
                    cell_content.append('')
            except Exception as exc:
                print(
                    f"Exception occurred while processing path=[{self.options['path']}], "
                    f"cell=[{self.options['cells']}]]\n{exc}"
                )

        node = CodeBlock(
            content=cell_content,
            name=self.name,
            arguments=self.arguments,
            options=self.options,
            lineno=self.lineno,
            content_offset=self.content_offset,
            block_text=self.block_text,
            state=self.state,
            state_machine=self.state_machine,
        )
        node = node.run()[0]
        return [node]


def _add_prompts(content_line: str) -> str:
    if content_line == '':
        return content_line
    elif content_line.startswith('    '):
        return '... ' + content_line
    else:
        return '>>> ' + content_line


def setup(app):
    app.add_directive("nbimport", NbImport)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
