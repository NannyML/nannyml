#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing metric utilities and implementations."""
import abc
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from nannyml.exceptions import InvalidArgumentsException
from nannyml.metadata import NML_METADATA_PREDICTION_COLUMN_NAME, NML_METADATA_TARGET_COLUMN_NAME


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
        calculation_function : Callable
            A function that will calculate a performance metric.
        upper_threshold : float, default=None
            An optional upper threshold for the performance metric.
        lower_threshold : float, default=None
            An optional lower threshold for the performance metric.
        """
        self.display_name = display_name
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def calculate(self, data: pd.DataFrame):
        if NML_METADATA_TARGET_COLUMN_NAME not in data.columns:
            raise RuntimeError('data does not contain target column')

        if NML_METADATA_PREDICTION_COLUMN_NAME not in data.columns:
            raise RuntimeError('data does contains neither prediction column or predicted probabilities column')

        return self._calculate(data)

    def _calculate(self, data: pd.DataFrame):
        raise NotImplementedError

    def __eq__(self, other):
        """Establishes equality by comparing all properties."""
        return (
            self.display_name == other.display_name
            and self.upper_threshold == other.upper_threshold
            and self.lower_threshold == other.lower_threshold
        )


class AUROC(Metric):
    """Area under Receiver Operating Curve metric."""

    def __init__(self):
        """Creates a new AUROC instance."""
        super().__init__(display_name='ROC_AUC')

    def _calculate(self, data: pd.DataFrame):
        """
        Redefine to handle NaNs and edge cases.
        """

        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[NML_METADATA_PREDICTION_COLUMN_NAME]  # TODO: this should be predicted_probabilities

        _common_data_cleaning(y_true, y_pred)

        if y_true.nunique() <= 1:
            return np.nan
        else:
            return roc_auc_score(y_true, y_pred)


class F1(Metric):
    """F1 score metric."""

    def __init__(self):
        """Creates a new F1 instance."""
        super().__init__(display_name='F1')

    def _calculate(self, data: pd.DataFrame):
        """
        Redefine to handle NaNs and edge cases.
        """
        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[NML_METADATA_PREDICTION_COLUMN_NAME]

        _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return f1_score(y_true, y_pred)


class Precision(Metric):
    """Precision metric."""

    def __init__(self):
        super().__init__(display_name='precision')

    def _calculate(self, data: pd.DataFrame):
        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[NML_METADATA_PREDICTION_COLUMN_NAME]

        _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return precision_score(y_true, y_pred)


class Recall(Metric):
    def __init__(self):
        super().__init__(display_name='recall')

    def _calculate(self, data: pd.DataFrame):
        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[NML_METADATA_PREDICTION_COLUMN_NAME]

        _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return recall_score(y_true, y_pred)


class Specificity(Metric):
    def __init__(self):
        super().__init__(display_name='specificity')

    def _calculate(self, data: pd.DataFrame):
        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[NML_METADATA_PREDICTION_COLUMN_NAME]

        _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fp)


class Sensitivity(Metric):
    def __init__(self):
        super().__init__(display_name='sensitivity')

    def _calculate(self, data: pd.DataFrame):
        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[NML_METADATA_PREDICTION_COLUMN_NAME]

        _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tp / (tp + fn)


class Accuracy(Metric):
    def __init__(self):
        super().__init__(display_name='accuracy')

    def _calculate(self, data: pd.DataFrame):
        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[NML_METADATA_PREDICTION_COLUMN_NAME]

        _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tp + tn / (tp + tn + fp + fn)


def _common_data_cleaning(y_true, y_pred):
    y_true, y_pred = (
        pd.Series(y_true).reset_index(drop=True),
        pd.Series(y_pred).reset_index(drop=True),
    )
    y_true = y_true[~y_pred.isna()]
    y_pred.dropna(inplace=True)

    y_pred = y_pred[~y_true.isna()]
    y_true.dropna(inplace=True)


class MetricFactory:
    """A factory class that produces Metric instances based on a given magic string or a metric calculation function."""

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
