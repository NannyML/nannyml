#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any, Dict

import fsspec
import pandas as pd

from nannyml.exceptions import InvalidArgumentsException
from nannyml.io.base import Reader, get_filepath_str, get_protocol_and_path


class FileReader(Reader):
    def __init__(
        self,
        filepath: str,
        read_args: Dict[str, Any] = None,
        credentials: Dict[str, Any] = None,
        fs_args: Dict[str, Any] = None,
    ):
        _fs_args = deepcopy(fs_args) or {}
        _credentials = deepcopy(credentials) or {}

        protocol, path = get_protocol_and_path(filepath)
        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)

        self._protocol = protocol
        self._storage_options = {**_credentials, **_fs_args}
        self._fs = fsspec.filesystem(self._protocol, **self._storage_options)
        self._filepath = PurePosixPath(path)

        self._read_args = read_args or {}  # type: Dict[str, Any]

    def _read(self) -> pd.DataFrame:
        read_path = get_filepath_str(self._filepath, self._protocol)

        with self._fs.open(read_path, mode='rb') as f:
            if self._filepath.suffix in ['.pq', '.parquet']:
                return pd.read_parquet(f, **self._read_args)
            elif self._filepath.suffix == '.csv':
                return pd.read_csv(f, **self._read_args)
            else:
                raise InvalidArgumentsException(f"'{self._filepath.suffix}' files are currently not supported.")
