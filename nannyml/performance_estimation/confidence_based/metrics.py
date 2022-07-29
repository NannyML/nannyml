import abc
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

from nannyml.base import AbstractEstimator
from nannyml.chunk import Chunker, Chunk


class Metric(abc.ABC):
    """A performance metric used to calculate realized model performance."""

    def __init__(
        self,
        display_name: str,
        column_name: str,
        estimator: AbstractEstimator,
    ):
        """Creates a new Metric instance.

        Parameters
        ----------
        display_name : str
            The name of the metric. Used to display in plots. If not given this name will be derived from the
            ``calculation_function``.
        column_name: str
            The name used to indicate the metric in columns of a DataFrame.
        """
        self.display_name = display_name
        self.column_name = column_name

        from .cbpe import CBPE
        if not isinstance(estimator, CBPE):
            raise RuntimeError(f"{estimator.__class__.__name__} is not an instance of type " f"CBPE")

        self.estimator = estimator

        self.upper_threshold: Optional[float] = None
        self.lower_threshold: Optional[float] = None

        self.reference_stability = 0.0

    def fit(self, reference_data: pd.DataFrame, chunker: Chunker):
        """Fits a Metric on reference data.

        Parameters
        ----------
        reference_data: pd.DataFrame
            The reference data used for fitting. Must have target data available.
        chunker: Chunker
            The :class:`~nannyml.chunk.Chunker` used to split the reference data into chunks.

        """
        self._fit(reference_data)

        # Calculate alert thresholds
        reference_chunks = chunker.split(
            reference_data,
            timestamp_column_name=self.estimator.timestamp_column_name,
        )
        self.lower_threshold, self.upper_threshold = self._alert_thresholds(reference_chunks)

        return

    @abc.abstractmethod
    def _fit(self, reference_data: pd.DataFrame):
        raise NotImplementedError

    def estimate(self, data: pd.DataFrame):
        """Calculates performance metrics on data.

        Parameters
        ----------
        data: pd.DataFrame
            The data to calculate performance metrics on. Requires presence of either the predicted labels or
            prediction scores/probabilities (depending on the metric to be calculated), as well as the target data.
        """
        reference_chunks = self.estimator.chunker.split(data, self.estimator.timestamp_column_name)
        return self._estimate(reference_chunks)

    @abc.abstractmethod
    def _estimate(self, reference_chunks: List[Chunk]):
        raise NotImplementedError

    def _confidence_bounds(self, reference_chunks: List[Chunk]):
        confidence_deviation = np.std([self._estimate(chunk.data) for chunk in reference_chunks])

    def _alert_thresholds(
        self, reference_chunks: List[Chunk], std_num: int = 3, lower_limit: int = 0, upper_limit: int = 1
    ) -> Tuple[float, float]:
        realized_chunk_performance = [
            self._realized_performance(chunk, self.estimator.y_true, self.estimator.y_pred, self.estimator.y_pred_proba)
            for chunk in reference_chunks
        ]
        deviation = np.std(realized_chunk_performance) * std_num
        mean_realised_performance = np.mean(realized_chunk_performance)
        lower_threshold = np.maximum(mean_realised_performance - deviation, lower_limit)
        upper_threshold = np.minimum(mean_realised_performance + deviation, upper_limit)

        return lower_threshold, upper_threshold

    @abc.abstractmethod
    def _realized_performance(self, chunk: Chunk, y_true_col: str, y_pred_col: str,
                              y_pred_proba_col: str) -> float:
        raise NotImplementedError

    def __eq__(self, other):
        """Establishes equality by comparing all properties."""
        return (
            self.display_name == other.display_name
            and self.column_name == other.column_name
        )


class BinaryClassificationAUROC(Metric):

    def _fit(self, reference_data: pd.DataFrame):
        pass

    def _estimate(self, data: pd.DataFrame):
        pass

    def _realized_performance(self, chunk: Chunk, y_true_col: str, y_pred_col: str, y_pred_proba_col: str) -> float:
        pass
