#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import fsspec
import joblib

from nannyml._typing import Calculator, Estimator
from nannyml.exceptions import CalculatorStoreLoadException
from nannyml.io.base import _get_protocol_and_path, get_filepath_str
from nannyml.io.calculator_store.base import Store


class FilesystemStore(Store):
    def __init__(
        self,
        root_path: str,
        store_args: Optional[Dict[str, Any]] = None,
        credentials: Optional[Dict[str, Any]] = None,
        fs_args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        _fs_args = deepcopy(fs_args) or {}
        _credentials = deepcopy(credentials) or {}

        protocol, path = _get_protocol_and_path(root_path)
        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)

        self._protocol = protocol
        self.root_path = path
        self._storage_options = {**_credentials, **_fs_args}
        self._fs = fsspec.filesystem(self._protocol, **self._storage_options)

        self._store_args = store_args or {}  # type: Dict[str, Any]

    def store(self, calculator: Union[Calculator, Estimator], path: str = '', filename: Optional[str] = None):
        if not filename:
            filename = f'{calculator.__module__}.{calculator.__class__.__name__}.pkl'

        write_path = Path(get_filepath_str(self.root_path, self._protocol)) / path / filename

        with self._fs.open(str(write_path), mode="wb") as fs_file:
            joblib.dump(calculator, fs_file)

    def load(self, filename: str, **load_args) -> Union[Calculator, Estimator]:
        path = load_args.get('path', '')

        try:
            load_path = Path(get_filepath_str(self.root_path, self._protocol)) / path / filename
            with self._fs.open(str(load_path), mode="rb") as fs_file:
                calc = joblib.load(fs_file)
                return calc
        except Exception as exc:
            raise CalculatorStoreLoadException(f'{exc}')
