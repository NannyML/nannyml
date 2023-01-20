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
    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __str__(self):
        return f'{self.__module__}.{self.__class__.__name__}'

    def serialize(self, obj, *args, **kwargs) -> bytes:
        try:
            self._logger.debug(f'serializing object {obj}')
            return self._serialize(obj, args, kwargs)
        except Exception as exc:
            raise SerializeException(f'an unexpected exception occurred when serializing "{obj}": {exc}')

    @abstractmethod
    def _serialize(self, obj, *args, **kwargs) -> bytes:
        ...

    def deserialize(self, bytez: bytes, *args, **kwargs) -> object:
        try:
            self._logger.debug('deserializing bytes')
            return self._deserialize(bytez, args, kwargs)
        except Exception as exc:
            raise SerializeException(f'an unexpected exception occurred when deserializing: {exc}')

    @abstractmethod
    def _deserialize(self, bytez: bytes, *args, **kwargs) -> object:
        ...


class PickleSerializer(Serializer):
    def _serialize(self, obj, *args, **kwargs) -> bytes:
        return pickle.dumps(obj)

    def _deserialize(self, bytez: bytes, *args, **kwargs) -> object:
        return pickle.loads(bytez)


class JoblibPickleSerializer(Serializer):
    def _serialize(self, obj, *args, **kwargs) -> bytes:
        b = BytesIO()
        joblib.dump(obj, b)
        b.seek(0)
        bytez = b.read()

        return bytez

    def _deserialize(self, bytez: bytes, *args, **kwargs) -> object:
        b = BytesIO(bytez)
        return joblib.load(b)
