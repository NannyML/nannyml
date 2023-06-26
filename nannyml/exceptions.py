# Author:   Niels Nuyttens  <niels@nannyml.com>
#
# License: Apache Software License 2.0

"""Custom exceptions."""


class NannyMLException(Exception):
    """Base class for all NannyML exceptions."""


class InvalidArgumentsException(NannyMLException):
    """An exception indicating that the inputs for a function are invalid."""


class ChunkerException(NannyMLException):
    """An exception indicating an error occurred somewhere during chunking."""


class MissingMetadataException(NannyMLException):
    """An exception indicating metadata columns are missing from the dataframe being processed."""


class InvalidReferenceDataException(NannyMLException):
    """An exception indicating the reference data provided are invalid."""


class CalculatorException(NannyMLException):
    """An exception indicating an error occurred during calculation."""


class EstimatorException(NannyMLException):
    """An exception indicating an error occurred during estimation."""


class CalculatorNotFittedException(CalculatorException):
    """An exception indicating a calculator was not fitted before using it in calculations."""


class NotFittedException(NannyMLException):
    """An exception indicating an object was not fitted before using it."""


class WriterException(NannyMLException):
    """An exception indicating something went wrong whilst trying to write out results."""


class ReaderException(NannyMLException):
    """An exception indicating something went wrong whilst trying to read out data."""


class IOException(NannyMLException):
    """An exception indicating something went wrong during IO."""


class StoreException(NannyMLException):
    """An exception indicating something went wrong whilst using a store."""


class SerializeException(NannyMLException):
    """An exception occurring when serialization some object went wrong."""


class DeserializeException(NannyMLException):
    """An exception occurring when deserialization some object went wrong."""


class ThresholdException(NannyMLException):
    """An exception occurring during threshold creation or calculation."""
