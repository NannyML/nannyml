# Author:   Niels Nuyttens  <niels@nannyml.com>
#
# License: Apache Software License 2.0

"""Custom exceptions."""


class InvalidArgumentsException(BaseException):
    """An exception indicating that the inputs for a function are invalid."""

    pass


class ChunkerException(BaseException):
    """An exception indicating an error occurred somewhere during chunking."""

    pass


class MissingMetadataException(BaseException):
    """An exception indicating metadata columns are missing from the dataframe being processed."""

    pass


class CalculatorException(BaseException):
    """An exception indicating an error occurred during (drift) calculation."""

    pass


class CalculatorNotFittedException(CalculatorException):
    """An exception indicating a calculator was not fitted before using it in calculations."""

    pass


class NotFittedException(BaseException):
    """An exception indicating an object was not fitted before using it."""

    pass
