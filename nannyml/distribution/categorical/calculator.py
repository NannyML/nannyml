from typing import Union, List, Optional

import pandas as pd
from typing_extensions import Self

from nannyml import Chunker
from nannyml.base import AbstractCalculator
from nannyml.distribution.categorical.result import Result


class CategoricalDistributionCalculator(AbstractCalculator):
    def __init__(
        self,
        column_names: Union[str, List[str]],
        timestamp_column_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_number: Optional[int] = None,
        chunk_period: Optional[str] = None,
        chunker: Optional[Chunker] = None,
    ):
        super().__init__(
            chunk_size,
            chunk_number,
            chunk_period,
            chunker,
            timestamp_column_name,
        )

        self.column_names = column_names
        self.result: Optional[Result] = None

    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> Self:
        pass

    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> Result:
        pass
