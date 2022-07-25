#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Implementation of the CBPE estimator."""
from abc import abstractmethod
from typing import Dict, List, Tuple

import pandas as pd

from nannyml._typing import ModelOutputsType, UseCase, derive_use_case
from nannyml.base import AbstractEstimator
from nannyml.calibration import Calibrator, CalibratorFactory
from nannyml.chunk import Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_estimation.confidence_based.results import (
    SUPPORTED_METRIC_VALUES,
    CBPEPerformanceEstimatorResult,
)


class CBPE(AbstractEstimator):
    """Performance estimator using the Confidence Based Performance Estimation (CBPE) technique."""

    def __new__(cls, y_pred_proba: ModelOutputsType, *args, **kwargs):
        """Creates a new CBPE subclass instance based on the type of the provided ``model_metadata``."""
        from ._cbpe_binary_classification import _BinaryClassificationCBPE
        from ._cbpe_multiclass_classification import _MulticlassClassificationCBPE

        use_case = derive_use_case(y_pred_proba)

        if use_case is UseCase.CLASSIFICATION_BINARY:
            return super(CBPE, cls).__new__(_BinaryClassificationCBPE)
        elif use_case is UseCase.CLASSIFICATION_MULTICLASS:
            return super(CBPE, cls).__new__(_MulticlassClassificationCBPE)
        else:
            raise NotImplementedError

    def __init__(
        self,
        metrics: List[str],
        y_pred: str,
        y_pred_proba: ModelOutputsType,
        y_true: str,
        timestamp_column_name: str,
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
        metrics: List[str]
            A list of metrics to calculate.
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
        >>>
        >>> reference_df, analysis_df, target_df = nml.load_synthetic_binary_classification_dataset()
        >>>
        >>> estimator = nml.CBPE(
        >>>     y_true='work_home_actual',
        >>>     y_pred='y_pred',
        >>>     y_pred_proba='y_pred_proba',
        >>>     timestamp_column_name='timestamp',
        >>>     metrics=['f1', 'roc_auc']
        >>> )
        >>>
        >>> estimator.fit(reference_df)
        >>>
        >>> results = estimator.estimate(analysis_df)
        >>> print(results.data)
                     key  start_index  ...  lower_threshold_roc_auc alert_roc_auc
        0       [0:4999]            0  ...                  0.97866         False
        1    [5000:9999]         5000  ...                  0.97866         False
        2  [10000:14999]        10000  ...                  0.97866         False
        3  [15000:19999]        15000  ...                  0.97866         False
        4  [20000:24999]        20000  ...                  0.97866         False
        5  [25000:29999]        25000  ...                  0.97866          True
        6  [30000:34999]        30000  ...                  0.97866          True
        7  [35000:39999]        35000  ...                  0.97866          True
        8  [40000:44999]        40000  ...                  0.97866          True
        9  [45000:49999]        45000  ...                  0.97866          True
        >>> for metric in estimator.metrics:
        >>>     results.plot(metric=metric, plot_reference=True).show()

        """
        super().__init__(chunk_size, chunk_number, chunk_period, chunker)

        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.timestamp_column_name = timestamp_column_name

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
    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> AbstractEstimator:
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
    def _estimate(self, data: pd.DataFrame, *args, **kwargs) -> CBPEPerformanceEstimatorResult:
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
