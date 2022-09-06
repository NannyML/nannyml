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
        _clear_image_outputs(nb)
    except CellExecutionError:
        print(f'Error executing the notebook "{nb_path}".\n\n')
        raise
    finally:
        nb_out_path = out_dir / Path(nb_path).name
        with open(nb_out_path, mode='w', encoding='utf-8') as f:
            nbformat.write(nb, f)


def _clear_image_outputs(nb: NotebookNode) -> NotebookNode:
    def is_image_output(cell_output: NotebookNode) -> bool:
        if 'data' in cell_output and 'application/vnd.plotly.v1+json' in cell_output['data']:
            return True
        return False

    images_found = 0

    for cell in nb['cells']:
        if 'outputs' in cell:
            for output in cell['outputs']:
                if is_image_output(output):
                    images_found += 1
                    cell['outputs'] = []

    sys.stdout.write(f' -- removed {images_found} image outputs')
    return nb


if __name__ == '__main__':
    print('=========================== running notebooks ===========================')
    notebooks_dir = sys.argv[1]
    for path in glob.iglob(notebooks_dir, recursive=True):
        s = time.time()
        sys.stdout.write('running ' + path)
        sys.stdout.flush()
        run_notebook(path)
        sys.stdout.write(' -- Finish in {}s\n'.format(int(time.time() - s)))

print('\n\033[92m' '===========================' ' Notebook testing done ' '===========================' '\033[0m')
