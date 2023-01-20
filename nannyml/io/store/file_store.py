#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import fsspec

from nannyml.io.base import _get_protocol_and_path, get_filepath_str
from nannyml.io.store.base import Store
from nannyml.io.store.serializers import JoblibPickleSerializer, Serializer


class FilesystemStore(Store):
    def __init__(
        self,
        root_path: str,
        store_args: Optional[Dict[str, Any]] = None,
        credentials: Optional[Dict[str, Any]] = None,
        fs_args: Optional[Dict[str, Any]] = None,
        serializer: Serializer = JoblibPickleSerializer(),
    ):
        super().__init__()

        _fs_args = deepcopy(fs_args) or {}
        _credentials = deepcopy(credentials) or {}

        protocol, path = _get_protocol_and_path(root_path)
        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)

        self._protocol = protocol.lower()
        self.root_path = path
        self._storage_options = {**_credentials, **_fs_args}
        self._fs = fsspec.filesystem(self._protocol, **self._storage_options)

        self._store_args = store_args or {}  # type: Dict[str, Any]

        self._serializer = serializer

    def _store(self, obj, path: Optional[str] = None, **store_args):
        if not path:
            path = f'{obj.__module__}.{obj.__class__.__name__}.pkl'

        write_path = Path(get_filepath_str(self.root_path, self._protocol)) / path

        with self._fs.open(str(write_path), mode="wb") as fs_file:
            bytez = self._serializer.serialize(obj)
            fs_file.write(bytez)

    def _load(self, path: str, **load_args):
        try:
            load_path = Path(get_filepath_str(self.root_path, self._protocol)) / path
            with self._fs.open(str(load_path), mode="rb") as fs_file:
                bytez = fs_file.read()
                calc = self._serializer.deserialize(bytez)
                return calc
        except FileNotFoundError:
            p = f'{self._protocol}://{self.root_path}/{path}'
            self._logger.warning(f'could not find file in location "{p}", returning "None"')
            return None
