#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing metric utilities and implementations."""
import abc
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import PolynomialFeatures

from nannyml import Chunk, Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.metadata import (
    NML_METADATA_PARTITION_COLUMN_NAME,
    NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME,
    NML_METADATA_PREDICTION_COLUMN_NAME,
    NML_METADATA_REFERENCE_PARTITION_NAME,
    NML_METADATA_TARGET_COLUMN_NAME,
)


class Metric(abc.ABC):
    """Represents a performance metric."""

    def __init__(
        self,
        display_name: str,
        upper_threshold: float = None,
        lower_threshold: float = None,
    ):
        """Creates a new Metric instance.

        Parameters
        ----------
        display_name : str
            The name of the metric. Used to display in plots. If not given this name will be derived from the
            ``calculation_function``.
        upper_threshold : float, default=None
            An optional upper threshold for the performance metric.
        lower_threshold : float, default=None
            An optional lower threshold for the performance metric.
        """
        self.display_name = display_name
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def fit(self, reference_data: pd.DataFrame, chunker: Chunker):
        """Fits a Metric on reference data.

        Parameters
        ----------
        reference_data: pd.DataFrame
            The reference data used for fitting. Must have target data available.
        chunker: Chunker
            The :class:`~nannyml.chunk.Chunker` used to split the reference data into chunks.
            This value is provided by the calling
            :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator`.

        """
        self._fit(reference_data)

        # Calculate alert thresholds
        if self.upper_threshold is None and self.lower_threshold is None:
            reference_chunks = chunker.split(reference_data, minimum_chunk_size=self.minimum_chunk_size())
            self.lower_threshold, self.upper_threshold = self._calculate_alert_thresholds(reference_chunks)

        return

    def _fit(self, reference_data: pd.DataFrame):
        raise NotImplementedError

    def calculate(self, data: pd.DataFrame):
        """Calculates performance metrics on data.

        Parameters
        ----------
        data: pd.DataFrame
            The data to calculate performance metrics on. Requires presence of either the predicted labels or
            prediction scores/probabilities (depending on the metric to be calculated), as well as the target data.
        """
        if NML_METADATA_TARGET_COLUMN_NAME not in data.columns:
            raise RuntimeError('data does not contain target column')

        if (
            NML_METADATA_PREDICTION_COLUMN_NAME not in data.columns
            and NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME not in data.columns
        ):
            raise RuntimeError('data does contains neither prediction column or predicted probabilities column')

        return self._calculate(data)

    def _calculate(self, data: pd.DataFrame):
        raise NotImplementedError

    def minimum_chunk_size(self) -> int:
        """Determines the minimum number of observations a chunk should ideally for this metric to be trustworthy."""
        return self._minimum_chunk_size()

    def _minimum_chunk_size(self):
        raise NotImplementedError

    def _calculate_alert_thresholds(
        self, reference_chunks: List[Chunk], std_num: int = 3, lower_limit: int = 0, upper_limit: int = 1
    ) -> Tuple[float, float]:
        chunked_reference_metric = [self.calculate(chunk.data) for chunk in reference_chunks]
        deviation = np.std(chunked_reference_metric) * std_num
        mean_reference_metric = np.mean(chunked_reference_metric)
        lower_threshold = np.maximum(mean_reference_metric - deviation, lower_limit)
        upper_threshold = np.minimum(mean_reference_metric + deviation, upper_limit)
        return lower_threshold, upper_threshold

    def __eq__(self, other):
        """Establishes equality by comparing all properties."""
        return (
            self.display_name == other.display_name
            and self.upper_threshold == other.upper_threshold
            and self.lower_threshold == other.lower_threshold
        )


def _minimum_roc_auc_based_chunk_size(
    data: pd.DataFrame,
    partition_column_name: str = NML_METADATA_PARTITION_COLUMN_NAME,
    predicted_probability_column_name: str = NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME,
    target_column_name: str = NML_METADATA_TARGET_COLUMN_NAME,
    lower_threshold: int = 300,
) -> int:
    def get_prediction(X):
        # model data
        h_coefs = [
            0.00000000e00,
            -3.46098897e04,
            2.65871679e04,
            3.46098897e04,
            2.29602791e04,
            -4.96886646e04,
            -1.12777343e-10,
            -2.29602791e04,
            3.13775672e-10,
            2.48718826e04,
        ]
        h_intercept = 1421.9522967076875
        transformation = PolynomialFeatures(3)
        #

        inputs = np.asarray(X)
        transformed_inputs = transformation.fit_transform(inputs)
        prediction = np.dot(transformed_inputs, h_coefs)[0] + h_intercept

        return prediction

    class_balance = np.mean(data[target_column_name])

    # Clean up NaN values
    y_true = data.loc[data[partition_column_name] == NML_METADATA_REFERENCE_PARTITION_NAME, target_column_name]
    y_pred_proba = data.loc[
        data[partition_column_name] == NML_METADATA_REFERENCE_PARTITION_NAME, predicted_probability_column_name
    ]

    y_true = y_true[~y_pred_proba.isna()]
    y_pred_proba.dropna(inplace=True)

    y_pred_proba = y_pred_proba[~y_true.isna()]
    y_true.dropna(inplace=True)

    auc = roc_auc_score(y_true=y_true, y_score=y_pred_proba)

    chunk_size = get_prediction([[class_balance, auc]])
    chunk_size = np.maximum(lower_threshold, chunk_size)
    chunk_size = np.round(chunk_size, -2)
    minimum_chunk_size = int(chunk_size)

    return minimum_chunk_size


class AUROC(Metric):
    """Area under Receiver Operating Curve metric."""

    def __init__(self):
        """Creates a new AUROC instance."""
        super().__init__(display_name='roc_auc')
        self._min_chunk_size = None

    def _minimum_chunk_size(self) -> int:
        return self._min_chunk_size

    def _fit(self, reference_data: pd.DataFrame):
        self._min_chunk_size = _minimum_roc_auc_based_chunk_size(reference_data)

    def _calculate(self, data: pd.DataFrame):
        """Redefine to handle NaNs and edge cases."""
        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME]  # TODO: this should be predicted_probabilities

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if y_true.nunique() <= 1:
            return np.nan
        else:
            return roc_auc_score(y_true, y_pred)


class F1(Metric):
    """F1 score metric."""

    def __init__(self):
        """Creates a new F1 instance."""
        super().__init__(display_name='F1')

    def _minimum_chunk_size(self) -> int:
        return 300

    def _fit(self, reference_data: pd.DataFrame):
        pass

    def _calculate(self, data: pd.DataFrame):
        """Redefine to handle NaNs and edge cases."""
        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[NML_METADATA_PREDICTION_COLUMN_NAME]

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return f1_score(y_true, y_pred)


class Precision(Metric):
    """Precision metric."""

    def __init__(self):
        """Creates a new Precision instance."""
        super().__init__(display_name='precision')

    def _minimum_chunk_size(self) -> int:
        return 300

    def _fit(self, reference_data: pd.DataFrame):
        pass

    def _calculate(self, data: pd.DataFrame):
        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[NML_METADATA_PREDICTION_COLUMN_NAME]

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return precision_score(y_true, y_pred)


class Recall(Metric):
    """Recall metric."""

    def __init__(self):
        """Creates a new Recall instance."""
        super().__init__(display_name='recall')

    def _minimum_chunk_size(self) -> int:
        return 300

    def _fit(self, reference_data: pd.DataFrame):
        pass

    def _calculate(self, data: pd.DataFrame):
        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[NML_METADATA_PREDICTION_COLUMN_NAME]

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return recall_score(y_true, y_pred)


class Specificity(Metric):
    """Specificity metric."""

    def __init__(self):
        """Creates a new F1 instance."""
        super().__init__(display_name='specificity')

    def _minimum_chunk_size(self) -> int:
        return 300

    def _fit(self, reference_data: pd.DataFrame):
        pass

    def _calculate(self, data: pd.DataFrame):
        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[NML_METADATA_PREDICTION_COLUMN_NAME]

        if y_pred.isna().all():
            raise InvalidArgumentsException(
                f"could not calculate metric {self.display_name}: " "prediction column contains no data"
            )

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fp)


class Sensitivity(Metric):
    """Sensitivity metric."""

    def __init__(self):
        """Creates a new Sensitivity instance."""
        super().__init__(display_name='sensitivity')

    def _minimum_chunk_size(self) -> int:
        return 300

    def _fit(self, reference_data: pd.DataFrame):
        pass

    def _calculate(self, data: pd.DataFrame):
        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[NML_METADATA_PREDICTION_COLUMN_NAME]

        if y_pred.isna().all():
            raise InvalidArgumentsException(
                f"could not calculate metric {self.display_name}: " "prediction column contains no data"
            )

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tp / (tp + fn)


class Accuracy(Metric):
    """Accuracy metric."""

    def __init__(self):
        """Creates a new Accuracy instance."""
        super().__init__(display_name='accuracy')

    def _minimum_chunk_size(self) -> int:
        return 300

    def _fit(self, reference_data: pd.DataFrame):
        pass

    def _calculate(self, data: pd.DataFrame):
        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[NML_METADATA_PREDICTION_COLUMN_NAME]

        if y_pred.isna().all():
            raise InvalidArgumentsException(
                f"could not calculate metric '{self.display_name}': " "prediction column contains no data"
            )

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return (tp + tn) / (tp + tn + fp + fn)


def _common_data_cleaning(y_true, y_pred):
    y_true, y_pred = (
        pd.Series(y_true).reset_index(drop=True),
        pd.Series(y_pred).reset_index(drop=True),
    )
    y_true = y_true[~y_pred.isna()]
    y_pred.dropna(inplace=True)

    y_pred = y_pred[~y_true.isna()]
    y_true.dropna(inplace=True)

    return y_true, y_pred


class MetricFactory:
    """A factory class that produces Metric instances based on a given magic string or a metric specification."""

    _str_to_metric: Dict[str, Metric] = {
        'roc_auc': AUROC(),
        'f1': F1(),
        'precision': Precision(),
        'recall': Recall(),
        'sensitivity': Sensitivity(),
        'specificity': Specificity(),
        'accuracy': Accuracy(),
    }

    @classmethod
    def create(cls, key: Union[str, Metric]) -> Metric:
        """Returns a Metric instance for a given key."""
        if isinstance(key, str):
            return cls._create_from_str(key)
        elif isinstance(key, Metric):
            return key
        else:
            raise InvalidArgumentsException(
                f"cannot create metric given a '{type(key)}'" "Please provide a string, function or Metric"
            )

    @classmethod
    def _create_from_str(cls, key: str) -> Metric:
        if key not in cls._str_to_metric:
            raise InvalidArgumentsException(f"unknown metric key '{key}' given. " "Should be one of ['roc_auc'].")
        return cls._str_to_metric[key]
