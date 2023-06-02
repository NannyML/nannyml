#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

import logging
from abc import ABC, abstractmethod
from typing import Optional

from nannyml.exceptions import StoreException


class Store(ABC):
    """The abstract class that serves as a base for concrete implementations.

    The Store is used to persist and retrieve Python objects at runtime.

    One example use case is storing calculators and estimators after fitting them to reference data, a potentially
    compute intensive operation. When the calculators and estimators are then used on analysis data repeatedly, they
    can be simply retrieved from the store, eliminating the need for repeated fitting.

    This abstract base class does not restrict in any way how the storage mechanism should work.

    The only implementation currently is the :class:`~nannyml.io.store.file_store.FilesystemStore`.
    """

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __str__(self):
        return f'{self.__module__}.{self.__class__.__name__}'

    def store(self, obj, **store_args):
        """Stores an object into the store, pass along any keyword argument to control this behavior.

        Parameters
        ----------
        obj : object
            The object to be stored
        store_args : Dict[str, Any]
            Additional arguments passed to the subclass `store` implementation

        Raises
        ------
        StoreException: occurs when an unexpected exception occurs during the store call.

        """
        try:
            self._logger.info(f'storing object "{obj}" to store "{self}"')
            return self._store(obj, **store_args)
        except Exception as exc:
            raise StoreException(f'an unexpected exception occurred when storing {obj}: {exc}')

    @abstractmethod
    def _store(self, obj, **store_args):
        """The method to be implemented by any implementing Store subclass."""
        ...

    def load(self, as_type: Optional[type] = None, **load_args):
        """Loads an object from the store, pass along any keyword argument to control this behavior.

        Parameters
        ----------
        as_type : Optional[type], default=None
            When provided the `load` method will check if the loaded object is an instance of `as_type` or `None`.
            If it is not a StoreException will be raised.

            The `None` will be returned when no calculator was present at the store location, for example
            when performing the first run of NannyML (nothing was cached yet at that point).

            The object will be returned unchecked when the `as_type` parameter is not provided.
        load_args : Dict[str, Any]
            Additional arguments passed to the subclass `store` implementation

        Returns
        -------
        obj: object

        """
        try:
            self._logger.info(f'loading object from store "{self}"')
            obj = self._load(**load_args)
            if as_type and obj and not isinstance(obj, as_type):
                raise StoreException(f'loaded object is not of type "{as_type}"')
            return obj
        except Exception as exc:
            raise StoreException(f'an unexpected exception occurred when loading object: {exc}')

    @abstractmethod
    def _load(self, **load_args):
        ...
