#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import abc
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import fsspec

from nannyml._typing import Result
from nannyml.io.base import Writer, get_protocol_and_path


class FileWriter(Writer, abc.ABC):

    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        path: str,
        format: str,
        write_args: Dict[str, Any] = None,
        credentials: Dict[str, Any] = None,
        fs_args: Dict[str, Any] = None,
    ):
        super().__init__()

        self.filepath = path

        _fs_args = deepcopy(fs_args) or {}
        _credentials = deepcopy(credentials) or {}
        self._data_format = format

        protocol, path = get_protocol_and_path(path)
        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)

        self._protocol = protocol
        self._storage_options = {**_credentials, **_fs_args}
        self._fs = fsspec.filesystem(self._protocol, **self._storage_options)

        self._write_args = write_args or {}  # type: Dict[str, Any]

    def _write(self, result: Result, **kwargs):
        raise NotImplementedError(
            f"'{self.__class__.__name__}' is a subclass of Writer and it must implement the _write method"
        )


def _write_bytes_to_filesystem(bytez, save_path: Path, fs: fsspec.spec.AbstractFileSystem):
    with fs.open(str(save_path), mode="wb") as fs_file:
        fs_file.write(bytez)
