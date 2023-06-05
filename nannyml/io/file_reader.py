#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any, Dict, Optional

import fsspec
import pandas as pd

from nannyml.exceptions import InvalidArgumentsException
from nannyml.io.base import Reader, _get_protocol_and_path, get_filepath_str


class FileReader(Reader):
    """A :class:`~nannyml.io.base.Reader` implementation that reads a local or cloud-based file."""

    def __init__(
        self,
        filepath: str,
        read_args: Optional[Dict[str, Any]] = None,
        credentials: Optional[Dict[str, Any]] = None,
        fs_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates a new ``FileReader`` instance.

        Parameters
        ----------
        filepath : str
            The path to read data from. Can be a regular file path or contain a protocol.
        read_args : Dict[str, Any], default=None
            Specific arguments passed along to the methods doing the actual reading (mostly Pandas-based).
        credentials : Dict[str, Any], default=None
            Used to provide credential information following specific ``fsspec`` implementations.
        fs_args : default=None
            Specific arguments passed along to the ``fsspec`` filesystem initializer.

        Examples
        --------
        >>> local_reader = FileReader(
        ...   filepath='/my-data-directory/data.pq'
        ... )

        >>> aws_reader = FileReader(
        ...   filepath='s3://my-data-directory/data.pq',
        ...   credentials={'key': 'my_key', 'secret': 'my_secret'}
        ... )

        >>> aws_reader2 = FileReader(
        ...   filepath='s3://my-data-directory/data.pq',
        ...   credentials={'aws_access_key_id': 'access_key_id', 'aws_secret_access_key': 'secret_access_key'}
        ... )

        >>> gcp_reader = FileReader(
        ...   filepath='s3://my-data-directory/data.pq',
        ...   credentials={'token': 'my_service_account_credential_file.json'}
        ... )

        """
        _fs_args = deepcopy(fs_args) or {}
        _credentials = deepcopy(credentials) or {}

        protocol, path = _get_protocol_and_path(filepath)
        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)

        self._protocol = protocol
        self._storage_options = {**_credentials, **_fs_args}
        self._fs = fsspec.filesystem(self._protocol, **self._storage_options)
        self._filepath = PurePosixPath(path)

        self._read_args = read_args or {}  # type: Dict[str, Any]

    def _read(self) -> pd.DataFrame:
        read_path = get_filepath_str(str(self._filepath), self._protocol)

        with self._fs.open(read_path, mode='rb') as f:
            if self._filepath.suffix in ['.pq', '.parquet']:
                return pd.read_parquet(f, **self._read_args)
            elif self._filepath.suffix == '.csv':
                return pd.read_csv(f, **self._read_args)
            else:
                raise InvalidArgumentsException(f"'{self._filepath.suffix}' files are currently not supported.")
