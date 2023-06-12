# Author:   Niels Nuyttens  <niels@nannyml.com>
#
# License: Apache Software License 2.0

"""Custom exceptions."""


class InvalidArgumentsException(Exception):
    """An exception indicating that the inputs for a function are invalid."""


class ChunkerException(Exception):
    """An exception indicating an error occurred somewhere during chunking."""


class MissingMetadataException(Exception):
    """An exception indicating metadata columns are missing from the dataframe being processed."""


class InvalidReferenceDataException(Exception):
    """An exception indicating the reference data provided are invalid."""


class CalculatorException(Exception):
    """An exception indicating an error occurred during calculation."""


class EstimatorException(Exception):
    """An exception indicating an error occurred during estimation."""


class CalculatorNotFittedException(CalculatorException):
    """An exception indicating a calculator was not fitted before using it in calculations."""


class NotFittedException(Exception):
    """An exception indicating an object was not fitted before using it."""


class WriterException(Exception):
    """An exception indicating something went wrong whilst trying to write out results."""


class ReaderException(Exception):
    """An exception indicating something went wrong whilst trying to read out data."""


class IOException(Exception):
    """An exception indicating something went wrong during IO."""


class StoreException(Exception):
    """An exception indicating something went wrong whilst using a store."""


class SerializeException(Exception):
    """An exception occurring when serialization some object went wrong."""


class DeserializeException(Exception):
    """An exception occurring when deserialization some object went wrong."""


class ThresholdException(Exception):
    """An exception occurring during threshold creation or calculation."""
