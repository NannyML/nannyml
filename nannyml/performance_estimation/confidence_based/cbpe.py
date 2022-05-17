#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Implementation of the CBPE estimator."""
from abc import abstractmethod
from typing import Dict, List, Tuple

import pandas as pd

from nannyml import MulticlassClassificationMetadata
from nannyml.calibration import Calibrator, CalibratorFactory
from nannyml.chunk import Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.metadata import BinaryClassificationMetadata, ModelMetadata
from nannyml.performance_estimation.base import PerformanceEstimator
from nannyml.performance_estimation.confidence_based.results import (
    SUPPORTED_METRIC_VALUES,
    CBPEPerformanceEstimatorResult,
)


class CBPE(PerformanceEstimator):
    """Performance estimator using the Confidence Based Performance Estimation (CBPE) technique."""

    def __new__(cls, model_metadata: ModelMetadata, *args, **kwargs):
        """Creates a new CBPE subclass instance based on the type of the provided ``model_metadata``."""
        from ._cbpe_binary_classification import _BinaryClassificationCBPE
        from ._cbpe_multiclass_classification import _MulticlassClassificationCBPE

        if isinstance(model_metadata, BinaryClassificationMetadata):
            return super(CBPE, cls).__new__(_BinaryClassificationCBPE)
        elif isinstance(model_metadata, MulticlassClassificationMetadata):
            return super(CBPE, cls).__new__(_MulticlassClassificationCBPE)
        else:
            raise NotImplementedError

    def __init__(
        self,
        model_metadata: ModelMetadata,
        metrics: List[str],
        features: List[str] = None,
        chunk_size: int = None,
        chunk_number: int = None,
        chunk_period: str = None,
        chunker: Chunker = None,
        calibration: str = None,
        calibrator: Calibrator = None,
    ):
        """Initializes a new CBPE performance estimator.

        Parameters
        ----------
        model_metadata: ModelMetadata
            Metadata telling the DriftCalculator what columns are required for drift calculation.
        metrics: List[str]
            A list of metrics to calculate.
        features: List[str], default=None
            An optional list of feature column names. When set only these columns will be included in the
            drift calculation. If not set all feature columns will be used.
        chunk_size: int, default=None
            Splits the data into chunks containing `chunks_size` observations.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunk_number: int, default=None
            Splits the data into `chunk_number` pieces.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunk_period: str, default=None
            Splits the data according to the given period.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunker : Chunker, default=None
            The `Chunker` used to split the data sets into a lists of chunks.
        calibration: str, default='isotonic'
            Determines which calibration will be applied to the model predictions. Defaults to ``isotonic``, currently
            the only supported value.
        calibrator: Calibrator, default=None
            A specific instance of a Calibrator to be applied to the model predictions.
            If not set NannyML will use the value of the ``calibration`` variable instead.

        Examples
        --------
        >>> import nannyml as nml
        >>> ref_df, ana_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>> metadata = nml.extract_metadata(ref_df)
        >>> # create a new estimator, chunking by week
        >>> estimator = nml.CBPE(model_metadata=metadata, chunk_period='W')

        """
        super().__init__(model_metadata, features, chunk_size, chunk_number, chunk_period, chunker)

        if metrics is None or len(metrics) == 0:
            raise InvalidArgumentsException(
                "no metrics provided. Please provide a non-empty list of metrics."
                f"Supported values are {SUPPORTED_METRIC_VALUES}."
            )

        for metric in metrics:
            if metric not in SUPPORTED_METRIC_VALUES:
                raise InvalidArgumentsException(
                    f"unknown 'metric' value: '{metric}'. " f"Supported values are {SUPPORTED_METRIC_VALUES}."
                )
        self.metrics = metrics

        self._confidence_deviations: Dict[str, float] = {}
        self._alert_thresholds: Dict[str, Tuple[float, float]] = {}
        self.needs_calibration: bool = False

        if calibrator is None:
            calibrator = CalibratorFactory.create(calibration)
        self.calibrator = calibrator

        self.minimum_chunk_size: int = None  # type: ignore

    @abstractmethod
    def fit(self, reference_data: pd.DataFrame) -> PerformanceEstimator:
        """Fits the drift calculator using a set of reference data.

        Parameters
        ----------
        reference_data : pd.DataFrame
            A reference data set containing predictions (labels and/or probabilities) and target values.

        Returns
        -------
        estimator: PerformanceEstimator
            The fitted estimator.

        Examples
        --------
        >>> import nannyml as nml
        >>> ref_df, ana_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>> metadata = nml.extract_metadata(ref_df, model_type=nml.ModelType.CLASSIFICATION_BINARY)
        >>> # create a new estimator and fit it on reference data
        >>> estimator = nml.CBPE(model_metadata=metadata, chunk_period='W').fit(ref_df)

        """
        pass

    @abstractmethod
    def estimate(self, data: pd.DataFrame) -> CBPEPerformanceEstimatorResult:
        """Calculates the data reconstruction drift for a given data set.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to calculate the reconstruction drift for.

        Returns
        -------
        estimates: PerformanceEstimatorResult
            A :class:`result<nannyml.performance_estimation.confidence_based.results.CBPEPerformanceEstimatorResult>`
            object where each row represents a :class:`~nannyml.chunk.Chunk`,
            containing :class:`~nannyml.chunk.Chunk` properties and the estimated metrics
            for that :class:`~nannyml.chunk.Chunk`.

        Examples
        --------
        >>> import nannyml as nml
        >>> ref_df, ana_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>> metadata = nml.extract_metadata(ref_df, model_type=nml.ModelType.CLASSIFICATION_BINARY)
        >>> # create a new estimator and fit it on reference data
        >>> estimator = nml.CBPE(model_metadata=metadata, chunk_period='W').fit(ref_df)
        >>> estimates = estimator.estimate(data)
        """
        pass
