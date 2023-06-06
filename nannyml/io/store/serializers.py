#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import logging
import pickle
from abc import ABC, abstractmethod
from io import BytesIO

import joblib

from nannyml.exceptions import SerializeException


class Serializer(ABC):
    """Abstract class for converting objects into bytes and the other way around."""

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __str__(self):
        return f'{self.__module__}.{self.__class__.__name__}'

    def serialize(self, obj, *args, **kwargs) -> bytes:
        """Convert an object into `bytes`.

        Parameters
        ----------
        obj : object
            The object to be converted.
        args : List[Any]
            Any arguments used to control the serialization.
        kwargs : Dict[str, Any]
            Any keyword arguments used to control the serialization.

        Returns
        -------
        bytez: bytes
            The byte representation of the given object

        Raises
        ------
        SerializeException: when an unexpected exception occurs during serialization.

        """
        try:
            self._logger.debug(f'serializing object {obj}')
            return self._serialize(obj, *args, **kwargs)
        except Exception as exc:
            raise SerializeException(f'an unexpected exception occurred when serializing "{obj}": {exc}')

    @abstractmethod
    def _serialize(self, obj, *args, **kwargs) -> bytes:
        ...

    def deserialize(self, bytez: bytes, *args, **kwargs) -> object:
        """Convert bytes back into an object

        Parameters
        ----------
        bytez : bytes
            The bytes to convert into an object
        args : List[Any]
            Any arguments used to control the deserialization.
        kwargs : Dict[str, Any]
            Any keyword arguments used to control the deserialization.

        Returns
        -------
        obj: object
            The object that was in the byte representation.

        """
        try:
            self._logger.debug('deserializing bytes')
            return self._deserialize(bytez, *args, **kwargs)
        except Exception as exc:
            raise SerializeException(f'an unexpected exception occurred when deserializing: {exc}')

    @abstractmethod
    def _deserialize(self, bytez: bytes, *args, **kwargs) -> object:
        ...


class PickleSerializer(Serializer):
    """A serializer based on the standard `pickle` library.

    Examples
    --------

    >>> ser = PickleSerializer()
    >>> b = ser.serialize(obj=calc)
    >>> loaded_calc = ser.deserialize(bytez=b)

    """

    def _serialize(self, obj, *args, **kwargs) -> bytes:
        return pickle.dumps(obj)

    def _deserialize(self, bytez: bytes, *args, **kwargs) -> object:
        return pickle.loads(bytez)


class JoblibPickleSerializer(Serializer):
    """A serializer based on the `joblib` pickling library.

    Examples
    --------

    >>> ser = JoblibPickleSerializer()
    >>> b = ser.serialize(obj=calc)
    >>> loaded_calc = ser.deserialize(bytez=b)

    """

    def _serialize(self, obj, *args, **kwargs) -> bytes:
        b = BytesIO()
        joblib.dump(obj, b)
        b.seek(0)
        bytez = b.read()

        return bytez

    def _deserialize(self, bytez: bytes, *args, **kwargs) -> object:
        b = BytesIO(bytez)
        return joblib.load(b)
