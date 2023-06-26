#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
"""The `io` package enables importing and exporting NannyML objects such as calculators or results.

The `file_reader` and `file_writer` modules contain base classes to read and write files from local or cloud storage.

The `raw_files_writer` allows exporting `Result` objects to disk in CSV or Parquet format.
The `pickle_file_writer` allows exporting `Result` objects to disk serialized using Python `pickle`.

The `db` module implements exporting `Result` objects to a database.

The `store` module implements an object cache, meant to cache fitted calculators in between runs.
"""

from .base import Reader, Writer, WriterFactory
from .db import DatabaseWriter
from .file_reader import FileReader
from .file_writer import FileWriter
from .pickle_file_writer import PickleFileWriter
from .raw_files_writer import RawFilesWriter
from .store import FilesystemStore, JoblibPickleSerializer, Serializer, Store

DEFAULT_WRITER = RawFilesWriter(path='out')
