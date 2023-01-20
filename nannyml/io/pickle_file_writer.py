#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

from nannyml._typing import Result
from nannyml.io.base import WriterFactory, get_filepath_str
from nannyml.io.file_writer import FileWriter, _write_bytes_to_filesystem
from nannyml.usage_logging import UsageEvent, log_usage


@WriterFactory.register('pickle')
class PickleFileWriter(FileWriter):
    """A Writer implementation that writes Results to disk (local/remote/cloud) as a pickle file."""

    def __init__(
        self,
        path: str,
        credentials: Optional[Dict[str, Any]] = None,
        fs_args: Optional[Dict[str, Any]] = None,
        write_args: Optional[Dict[str, Any]] = None,
    ):
        """

        Parameters
        ----------
        path : str
            The directory in which to output the generated pickle file. The name of the pickle file will equal
            the fully qualified result class name with a `pkl` extension, e.g. `nannyml.drift.univariate.result.pkl`
        credentials : Dict[str, Any]
            Used to provide credential information following specific ``fsspec`` implementations.
        fs_args :
            Specific arguments passed along to the ``fsspec`` filesystem initializer.

        Examples
        --------
        >>> writer = PickleFileWriter(
        ...    path='s3://my-output-bucket/output',
        ...    credentials={'aws_access_key_id': 'access_key_id', 'aws_secret_access_key': 'secret_access_key'}
        ... )
        >>> writer.write(result)
        """
        super().__init__(path, None, credentials, fs_args)

    @log_usage(UsageEvent.WRITE_PICKLE)
    def _write(self, result: Result, **kwargs):
        file_name = f'{result.__module__}.pkl'
        write_path = Path(get_filepath_str(self.filepath, self._protocol)) / file_name
        bytez = pickle.dumps(result)
        _write_bytes_to_filesystem(bytez, write_path, self._fs)
