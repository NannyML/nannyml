#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  #
#  License: Apache Software License 2.0

#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import logging
from copy import deepcopy
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Dict

import fsspec

from nannyml._typing import Result
from nannyml.exceptions import InvalidArgumentsException
from nannyml.io.base import Writer, get_filepath_str, get_protocol_and_path


class FileWriter(Writer):

    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        filepath: str,
        data_format: str,
        write_args: Dict[str, Any] = None,
        credentials: Dict[str, Any] = None,
        fs_args: Dict[str, Any] = None,
    ):
        _fs_args = deepcopy(fs_args) or {}
        _credentials = deepcopy(credentials) or {}
        self._data_format = data_format

        protocol, path = get_protocol_and_path(filepath)
        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)

        self._protocol = protocol
        self._storage_options = {**_credentials, **_fs_args}
        self._fs = fsspec.filesystem(self._protocol, **self._storage_options)

        super().__init__(filepath=PurePosixPath(path))

        self._write_args = write_args or {}  # type: Dict[str, Any]

    def _write(self, result: Result):
        write_path = get_filepath_str(self._filepath, self._protocol)

        images_path = Path(write_path) / result.calculator_name / "images"
        images_path.mkdir(parents=True, exist_ok=True)
        plots = result.plots.items()
        self._logger.debug(f"writing {len(plots)} images to {images_path}")
        for name, image in plots:
            _write_bytes_to_filesystem(image.to_image(format='png'), images_path / f'{name}.png', self._fs)

        data_path = Path(write_path) / result.calculator_name / "data"
        data_path.mkdir(parents=True, exist_ok=True)
        self._logger.debug(f"writing data to {data_path}")

        bytes_buffer = BytesIO()
        if self._data_format == "parquet":
            result.data.to_parquet(bytes_buffer, **self._write_args)
            _write_bytes_to_filesystem(bytes_buffer.getvalue(), data_path / f"{result.calculator_name}.pq", self._fs)
        elif self._data_format == "csv":
            result.data.to_csv(bytes_buffer, **self._write_args)
            _write_bytes_to_filesystem(bytes_buffer.getvalue(), data_path / f"{result.calculator_name}.csv", self._fs)
        else:
            raise InvalidArgumentsException(f"unknown value for format '{format}', should be one of 'parquet', 'csv'")


def _write_bytes_to_filesystem(bytez, save_path: Path, fs: fsspec.spec.AbstractFileSystem):
    with fs.open(str(save_path), mode="wb") as fs_file:
        fs_file.write(bytez)
