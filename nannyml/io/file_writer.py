#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import abc
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import fsspec

from nannyml._typing import Result
from nannyml.io.base import Writer, _get_protocol_and_path


class FileWriter(Writer, abc.ABC):
    """An abstract Writer implementation that writes results out to a filesystem (either local or remote / cloud)."""

    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        path: str,
        write_args: Optional[Dict[str, Any]] = None,
        credentials: Optional[Dict[str, Any]] = None,
        fs_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates a new ``FileWriter``

        Parameters
        ----------
        path : str
            The path to read data from. Can be a regular file path or contain a protocol.
        write_args : Dict[str, Any], default=None
            Specific arguments passed along the method performing the actual writing.
        credentials : Dict[str, Any], default=None
            Used to provide credential information following specific ``fsspec`` implementations.
        fs_args : default=None
            Specific arguments passed along to the ``fsspec`` filesystem initializer.
        """
        super().__init__()

        _fs_args = deepcopy(fs_args) or {}
        _credentials = deepcopy(credentials) or {}

        protocol, path = _get_protocol_and_path(path)
        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)

        self._protocol = protocol
        self.filepath = path
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
