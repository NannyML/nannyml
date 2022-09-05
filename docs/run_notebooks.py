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

ep = nbconvert.preprocessors.ExecutePreprocessor(
    extra_arguments=["--log-level=40"],
    timeout=300,
    kernel_name='python3',
)

cp = nbconvert.preprocessors.ClearOutputPreprocessor()

out_dir = pathlib.Path('docs/_build/notebooks')
out_dir.mkdir(parents=True, exist_ok=True)


def run_notebook(nb_path):
    nb_path = os.path.abspath(nb_path)
    assert path.endswith('.ipynb')
    nb = nbformat.read(path, as_version=4)
    try:
        cp.preprocess(nb, {'metadata': {'path': os.path.dirname(nb_path)}})
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(nb_path)}})
    except CellExecutionError:
        print(f'Error executing the notebook "{nb_path}".\n\n')
        raise
    finally:
        nb_out_path = out_dir / Path(nb_path).name
        with open(nb_out_path, mode='w', encoding='utf-8') as f:
            nbformat.write(nb, f)


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
