#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  #
#  License: Apache Software License 2.0

#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

import glob
import os
import pathlib
import sys
import time
from pathlib import Path

import nbconvert
import nbformat
from nbclient.exceptions import CellExecutionError
from nbformat import NotebookNode

NOTEBOOKS_TO_SKIP = ["Datasets - Census Employment MA"]

ep = nbconvert.preprocessors.ExecutePreprocessor(
    extra_arguments=["--log-level=40"],
    timeout=300,
    kernel_name='python3',
)

cp = nbconvert.preprocessors.ClearOutputPreprocessor()

out_dir = pathlib.Path('docs/example_notebooks')
out_dir.mkdir(parents=True, exist_ok=True)


def run_notebook(nb_path):
    nb_path = os.path.abspath(nb_path)
    assert path.endswith('.ipynb')
    nb = nbformat.read(path, as_version=4)
    try:
        cp.preprocess(nb, {'metadata': {'path': os.path.dirname(nb_path)}})
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(nb_path)}})
        postprocess(nb)
    except CellExecutionError:
        print(f'Error executing the notebook "{nb_path}".\n\n')
        raise
    finally:
        nb_out_path = out_dir / Path(nb_path).name
        with open(nb_out_path, mode='w', encoding='utf-8') as f:
            nbformat.write(nb, f)


def postprocess(nb: NotebookNode) -> NotebookNode:
    res = _clean_outputs(nb)
    res = _clear_execution_metadata(res)
    return res


def _clear_execution_metadata(nb: NotebookNode) -> NotebookNode:
    for cell in nb['cells']:
        if 'execution_count' in cell:
            cell['execution_count'] = None
        if 'metadata' in cell:
            if 'execution' in cell['metadata']:
                del cell['metadata']['execution']
            if 'pycharm' in cell['metadata']:
                del cell['metadata']['pycharm']
    return nb


def _clean_outputs(nb: NotebookNode) -> NotebookNode:
    def is_image_output(cell_output: NotebookNode) -> bool:
        if 'data' in cell_output and 'application/vnd.plotly.v1+json' in cell_output['data']:
            return True
        return False

    def is_nannyml_object_output(cell_output: NotebookNode) -> bool:
        if 'data' in cell_output and 'text/plain' in cell_output['data']:
            return str(cell_output['data']['text/plain']).startswith('<nannyml.')
        return False

    images_found = 0

    for cell in nb['cells']:
        if 'outputs' in cell:
            for output in cell['outputs']:
                if is_image_output(output):
                    images_found += 1
                    cell['outputs'] = []
                if is_nannyml_object_output(output):
                    cell['outputs'] = []

    sys.stdout.write(f' -- removed {images_found} image outputs')
    return nb


def skip_notebook(notebook_path):
    for notebook_name in NOTEBOOKS_TO_SKIP:
        if notebook_name in notebook_path:
            return True
    return False


if __name__ == '__main__':
    print('=========================== running notebooks ===========================')
    notebooks_dir = sys.argv[1]
    for path in glob.iglob(notebooks_dir, recursive=True):
        if skip_notebook(path):
            print('skipping ' + path)
        else:
            s = time.time()
            sys.stdout.write('running ' + path)
            sys.stdout.flush()
            run_notebook(path)
            sys.stdout.write(' -- Finish in {}s\n'.format(int(time.time() - s)))

print('\n\033[92m' '===========================' ' Notebook testing done ' '===========================' '\033[0m')
