#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import pickle
from pathlib import Path, PurePosixPath
from typing import Any, Dict

from nannyml._typing import Result
from nannyml.io.base import WriterFactory, get_filepath_str
from nannyml.io.file_writer import FileWriter, _write_bytes_to_filesystem


@WriterFactory.register('pickle')
class PickleFileWriter(FileWriter):
    def __init__(
        self,
        path: str,
        write_args: Dict[str, Any] = None,
        credentials: Dict[str, Any] = None,
        fs_args: Dict[str, Any] = None,
    ):
        super().__init__(path, write_args, credentials, fs_args)

    def _write(self, result: Result, **kwargs):
        file_name = f'{result.__module__}.pkl'
        write_path = Path(get_filepath_str(PurePosixPath(self.filepath), self._protocol)) / file_name
        bytez = pickle.dumps(result)
        _write_bytes_to_filesystem(bytez, write_path, self._fs)
