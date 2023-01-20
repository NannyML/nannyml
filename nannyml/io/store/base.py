#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import logging
from abc import ABC, abstractmethod
from typing import Optional

from nannyml.exceptions import StoreException


class Store(ABC):
    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __str__(self):
        return f'{self.__module__}.{self.__class__.__name__}'

    def store(self, obj, **store_args):
        try:
            self._logger.info(f'storing object "{obj}" to store "{self}"')
            return self._store(obj, **store_args)
        except Exception as exc:
            raise StoreException(f'an unexpected exception occurred when storing {obj}: {exc}')

    @abstractmethod
    def _store(self, obj, **store_args):
        ...

    def load(self, as_type: Optional[type] = None, **load_args):
        try:
            self._logger.info(f'loading object from store "{self}"')
            obj = self._load(**load_args)
            if as_type and not (obj is None or isinstance(obj, as_type)):
                raise StoreException(f'loaded object is not of type "{type}"')
            return obj
        except Exception as exc:
            raise StoreException(f'an unexpected exception occurred when loading object: {exc}')

    @abstractmethod
    def _load(self, path: str, **load_args):
        ...
