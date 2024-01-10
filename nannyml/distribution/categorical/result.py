from typing import List, Optional

import pandas as pd
from plotly.graph_objs import Figure

from nannyml import Chunker
from nannyml._typing import Key
from nannyml.base import PerColumnResult


class Result(PerColumnResult):
    def __init__(
        self,
        results_data: pd.DataFrame,
        column_names: List[str],
        timestamp_column_name: Optional[str],
        chunker: Chunker,
    ):
        super().__init__(results_data, column_names)

        self.timestamp_column_name = timestamp_column_name
        self.chunker = chunker

    def keys(self) -> List[Key]:
        pass

    def plot(self, *args, **kwargs) -> Figure:
        pass
