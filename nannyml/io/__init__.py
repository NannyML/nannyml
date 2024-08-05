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
from importlib import import_module

from .base import Reader, Writer, WriterFactory
from .file_reader import FileReader
from .file_writer import FileWriter
from .pickle_file_writer import PickleFileWriter
from .raw_files_writer import RawFilesWriter
from .store import FilesystemStore, JoblibPickleSerializer, Serializer, Store


_optional_dependencies = {
    'DatabaseWriter': '.db'
}


def __getattr__(name):
    optional_module_path = _optional_dependencies.get(name)
    if optional_module_path is not None:
        module = import_module(optional_module_path, package=__name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


DEFAULT_WRITER = RawFilesWriter(path='out')
