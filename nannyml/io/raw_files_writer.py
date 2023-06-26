#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

from nannyml._typing import Result
from nannyml.exceptions import InvalidArgumentsException
from nannyml.io.base import WriterFactory, get_filepath_str
from nannyml.io.file_writer import FileWriter, _write_bytes_to_filesystem
from nannyml.usage_logging import UsageEvent, log_usage


@WriterFactory.register('raw_files')
class RawFilesWriter(FileWriter):
    """Writes `Result` data and plots to disk (local/remote/cloud)."""

    def __init__(
        self,
        path: str,
        credentials: Optional[Dict[str, Any]] = None,
        fs_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates a new ``RawFilesWriter`` instance.

        Parameters
        ----------
        path : str
            The directory to write the results in.
            Each ``Result`` being written there will end create its own subdirectory.
            Each of those will contain `data` and `plots` subdirectories.
        format: str
            The file format for the data export. Should be one of ``parquet`` or ``csv``.
        write_args : Dict[str, Any], default=None
            Specific arguments passed along the method performing the actual writing.
        credentials : Dict[str, Any], default=None
            Used to provide credential information following specific ``fsspec`` implementations.
        fs_args : default=None
            Specific arguments passed along to the ``fsspec`` filesystem initializer.

        Examples
        --------
        >>> writer = RawFilesWriter(path='/output', format="parquet")
        >>> # plots is a Dictionary mapping a plot name to a plotly Figure
        >>> # this is some legacy stuff to be cleaned up
        >>> writer.write(result, plots={}, calculator_name='test')
        """
        super().__init__(path, credentials, fs_args)

    @log_usage(UsageEvent.WRITE_RAW)
    def _write(self, result: Result, format: str = 'parquet', **write_args):
        """Exports the Result data into a CSV or Parquet file on a disk location.

        The disk location is determined by the `path` given during initialization and the filename parameter.

        Parameters
        ----------
        result : Result
            The result to be exported
        filename : str
            The filename to use for the exported file.
        format: str
            The file format for the data export. Should be one of ``parquet`` or ``csv``.
        kwargs : dict
            A dictionary of key-value pairs passed to the function

        """
        if result.empty:
            raise InvalidArgumentsException("result data cannot be None")

        if format not in ['parquet', 'csv']:
            raise InvalidArgumentsException(f"unknown value for format '{format}', should be one of 'parquet', 'csv'")

        if 'filename' not in write_args:
            raise InvalidArgumentsException("missing required parameter 'filename'")
        filename = write_args.pop('filename')

        write_path = get_filepath_str(self.filepath, self._protocol)

        data_path = Path(write_path) / filename
        self._logger.debug(f"writing data to {data_path}")

        bytes_buffer = BytesIO()
        if format == "parquet":
            result.to_df(multilevel=False).to_parquet(
                bytes_buffer, coerce_timestamps='ms', allow_truncated_timestamps=True, **write_args
            )
            _write_bytes_to_filesystem(bytes_buffer.getvalue(), data_path, self._fs)
        elif format == "csv":
            result.to_df(multilevel=False).to_csv(bytes_buffer, **write_args)
            _write_bytes_to_filesystem(bytes_buffer.getvalue(), data_path, self._fs)
        else:
            raise InvalidArgumentsException(f"unknown value for format '{format}', should be one of 'parquet', 'csv'")
