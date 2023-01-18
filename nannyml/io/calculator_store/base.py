#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from abc import ABC, abstractmethod
from typing import Union

from nannyml._typing import Calculator, Estimator


class Store(ABC):
    @abstractmethod
    def store(self, calculator: Union[Calculator, Estimator], **store_args):
        ...

    @abstractmethod
    def load(self, **load_args) -> Union[Calculator, Estimator]:
        ...


class Serializer(ABC):
    @abstractmethod
    def serialize(self) -> bytearray:
        ...

    @abstractmethod
    def deserialize(self, bytez: bytearray) -> object:
        ...
