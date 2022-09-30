#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

from nannyml._typing import Result
from nannyml.exceptions import InvalidArgumentsException
from nannyml.io.base import WriterFactory, get_filepath_str
from nannyml.io.file_writer import FileWriter, _write_bytes_to_filesystem


@WriterFactory.register('raw_files')
class RawFilesWriter(FileWriter):
    def __init__(
        self,
        path: str,
        format: str,
        write_args: Dict[str, Any] = None,
        credentials: Dict[str, Any] = None,
        fs_args: Dict[str, Any] = None,
    ):
        super().__init__(path, write_args, credentials, fs_args)
        if format not in ['parquet', 'csv']:
            raise InvalidArgumentsException(f"unknown value for format '{format}', should be one of 'parquet', 'csv'")
        self._data_format = format

    def _write(self, result: Result, **kwargs):
        if 'plots' not in kwargs:
            raise InvalidArgumentsException("missing parameter 'plots'.")

        plots = kwargs['plots']

        if not isinstance(plots, Dict):
            raise InvalidArgumentsException(f"parameter 'plots' is of type {type(plots)} but should be 'Dict'")

        if result.data is None:
            raise InvalidArgumentsException("result data cannot be None")

        calculator_name = kwargs['calculator_name']
        write_path = get_filepath_str(Path(self.filepath), self._protocol)

        images_path = Path(write_path) / calculator_name / "plots"
        images_path.mkdir(parents=True, exist_ok=True)
        self._logger.debug(f"writing {len(plots)} images to {images_path}")
        for name, image in plots.items():
            if image is None:
                self._logger.debug(f"image for '{name}' is 'None'. Skipping writing to file.")
                break
            _write_bytes_to_filesystem(image.to_image(format='png'), images_path / f'{name}.png', self._fs)

        data_path = Path(write_path) / calculator_name / "data"
        data_path.mkdir(parents=True, exist_ok=True)
        self._logger.debug(f"writing data to {data_path}")

        bytes_buffer = BytesIO()
        if self._data_format == "parquet":
            result.data.to_parquet(bytes_buffer, **self._write_args)
            _write_bytes_to_filesystem(bytes_buffer.getvalue(), data_path / f"{calculator_name}.pq", self._fs)
        elif self._data_format == "csv":
            result.data.to_csv(bytes_buffer, **self._write_args)
            _write_bytes_to_filesystem(bytes_buffer.getvalue(), data_path / f"{calculator_name}.csv", self._fs)
        else:
            raise InvalidArgumentsException(f"unknown value for format '{format}', should be one of 'parquet', 'csv'")
