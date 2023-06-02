#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import fsspec

from nannyml.exceptions import InvalidArgumentsException
from nannyml.io.base import _get_protocol_and_path, get_filepath_str
from nannyml.io.store.base import Store
from nannyml.io.store.serializers import JoblibPickleSerializer, Serializer


class FilesystemStore(Store):
    """A Store implementation that uses a local or remote file system for persistence.

    Any object is first serialized using an instance of the :class:`~nannyml.io.store.serializers.Serializer` class.
    The resulting bytes are then written onto a file system.

    The ``FilesystemStore`` uses `fsspec` under the covers, allowing it to support a wide range of local and remote
    filesystems. These include (but are not limited to) S3, Google Cloud Storage and Azure Blob Storage.
    In order to these remote filesystems, additional credentials can be passed along.

    Examples
    ---------
    Using S3 as a backing filesystem.
    See `AWS documentation <https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html>`_ to
    learn more about the required access key id and secret access key credentials.

    >>> store = FilesystemStore(
    ...     root_path='s3://my-bucket-name/some/path',
    ...     credentials={
    ...         'client_kwargs': {
    ...            'aws_access_key_id': '<ACCESS_KEY_ID>'
    ...            'aws_secret_access_key': '<SECRET_ACCESS_KEY>'
    ...         }
    ...     }
    ... )

    Using Google Cloud Storage (GCS) as a backing filesystem.
    See `Google Cloud documentation <https://cloud.google.com/iam/docs/creating-managing-service-account-keys>`_
    to learn more about the required service account key credentials.

    >>> store = FilesystemStore(
    ...     root_path='gs://my-bucket-name/some/path',
    ...     credentials={'token': 'service-account-access-key.json'}
    ... )

    Using Azure Blob Storage as a backing filesystem.
    See `Azure support documentation <https://github.com/fsspec/adlfs#setting-credentials>`_ to learn more about
    the required credentials.

    >>> store = FilesystemStore(
    ...     root_path='abfs://my-container-name/some/path',
    ...     credentials={'account_name': '<ACCOUNT_NAME>', 'account_key': '<ACCOUNT_KEY>'}
    ... )

    Performing basic operations.
    An optional path parameter can be set to control what subdirectories and file name should be used when storing.
    When none is given the object will be stored in the configured store root path using an automatically generated
    name.

    >>> store = FilesystemStore(root_path='/tmp/nml-cache')  # creating the store
    >>> store.store(calc, path='example/calc.pkl')  # storing the object
    >>> store.load(path='example/calc.pkl')  # returns the object without any checks
    >>> # returns the object if it is a UnivariateDriftCalculator, raises a StoreException otherwise
    >>> store.load(path='example/calc.pkl', as_type='UnivariateDriftCalculator')
    >>> store.load(path='i_dont_exist.pkl')  # raises a StoreException

    """

    def __init__(
        self,
        root_path: str,
        credentials: Optional[Dict[str, Any]] = None,
        fs_args: Optional[Dict[str, Any]] = None,
        serializer: Serializer = JoblibPickleSerializer(),
    ):
        """Creates a new FilesystemStore instance.

        Parameters
        ----------
        root_path : str
            The root directory where all storage operations will originate.
        credentials : Optional[Dict[str, Any]], default=None
            Optional dictionary of credential information passed along to `fsspec`. Exact contents depend on the type
            of backing filesystem used.
        fs_args : Optional[Dict[str, Any]], default=None
            Optional dictionary of initialization parameters passed along when creating an internal `fsspec.filesystem`
            instance.
        serializer : Serializer, default=JoblibPickleSerializer()
            An optional :class:`~nannyml.io.store.serializers.Serializer` instance that will be used to convert
            an object into a byte representation and the other way around.
            The default uses the :class:`~nannyml.io.store.serializers.JoblibPickleSerializer`,
            which internally relies on *joblib* and it's pickling functionality.
        """
        super().__init__()

        _fs_args = deepcopy(fs_args) or {}
        _credentials = deepcopy(credentials) or {}

        protocol, path = _get_protocol_and_path(root_path)
        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)

        self._protocol = protocol.lower()
        self.path = path
        self._storage_options = {**_credentials, **_fs_args}
        self._fs = fsspec.filesystem(self._protocol, **self._storage_options)

        self._serializer = serializer

    def _store(self, obj, **store_args):
        if 'filename' not in store_args:
            raise InvalidArgumentsException("missing required parameter 'filename'")

        write_path = Path(get_filepath_str(self.path, self._protocol)) / store_args['filename']

        with self._fs.open(str(write_path), mode="wb") as fs_file:
            bytez = self._serializer.serialize(obj)
            fs_file.write(bytez)

    def _load(self, **load_args):
        if 'filename' not in load_args:
            raise InvalidArgumentsException("missing required parameter 'filename'")
        filename = load_args['filename']

        try:
            load_path = Path(get_filepath_str(self.path, self._protocol)) / filename
            with self._fs.open(str(load_path), mode="rb") as fs_file:
                bytez = fs_file.read()
                calc = self._serializer.deserialize(bytez)
                return calc
        except FileNotFoundError:
            p = f'{self._protocol}://{self.path}/{filename}'
            self._logger.info(f'could not find file in store location "{p}", returning "None"')
            return None
