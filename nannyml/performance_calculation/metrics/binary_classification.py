#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from nannyml._typing import ProblemType
from nannyml.base import _list_missing
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_calculation.metrics.base import Metric, MetricFactory, _common_data_cleaning
from nannyml.sampling_error.binary_classification import (
    accuracy_sampling_error,
    accuracy_sampling_error_components,
    auroc_sampling_error,
    auroc_sampling_error_components,
    f1_sampling_error,
    f1_sampling_error_components,
    precision_sampling_error,
    precision_sampling_error_components,
    recall_sampling_error,
    recall_sampling_error_components,
    specificity_sampling_error,
    specificity_sampling_error_components,
)


@MetricFactory.register(metric='roc_auc', use_case=ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationAUROC(Metric):
    """Area under Receiver Operating Curve metric."""

    def __init__(self, y_true: str, y_pred: str, y_pred_proba: Optional[str] = None):
        """Creates a new AUROC instance."""
        super().__init__(
            display_name='ROC AUC',
            column_name='roc_auc',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            lower_threshold_limit=0,
            upper_threshold_limit=1,
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def __str__(self):
        return "roc_auc"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred_proba], list(reference_data.columns))
        self._sampling_error_components = auroc_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_proba_reference=reference_data[self.y_pred_proba],
        )

    def _calculate(self, data: pd.DataFrame):
        """Redefine to handle NaNs and edge cases."""
        _list_missing([self.y_true, self.y_pred_proba], list(data.columns))

        y_true = data[self.y_true]
        y_pred = data[self.y_pred_proba]

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if y_true.nunique() <= 1:
            return np.nan
        else:
            return roc_auc_score(y_true, y_pred)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return auroc_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='f1', use_case=ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationF1(Metric):
    """F1 score metric."""

    def __init__(self, y_true: str, y_pred: str, y_pred_proba: Optional[str] = None):
        """Creates a new F1 instance."""
        super().__init__(
            display_name='F1',
            column_name='f1',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            lower_threshold_limit=0,
            upper_threshold_limit=1,
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def __str__(self):
        return "f1"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        self._sampling_error_components = f1_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
        )

    def _calculate(self, data: pd.DataFrame):
        """Redefine to handle NaNs and edge cases."""
        _list_missing([self.y_true, self.y_pred], list(data.columns))

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return f1_score(y_true, y_pred)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return f1_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='precision', use_case=ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationPrecision(Metric):
    """Precision metric."""

    def __init__(self, y_true: str, y_pred: str, y_pred_proba: Optional[str] = None):
        """Creates a new Precision instance."""
        super().__init__(
            display_name='Precision',
            column_name='precision',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            lower_threshold_limit=0,
            upper_threshold_limit=1,
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def __str__(self):
        return "precision"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        self._sampling_error_components = precision_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
        )

    def _calculate(self, data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(data.columns))

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return precision_score(y_true, y_pred)

    def _sampling_error(self, data: pd.DataFrame):
        return precision_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='recall', use_case=ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationRecall(Metric):
    """Recall metric, also known as 'sensitivity'."""

    def __init__(self, y_true: str, y_pred: str, y_pred_proba: Optional[str] = None):
        """Creates a new Recall instance."""
        super().__init__(
            display_name='Recall',
            column_name='recall',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            lower_threshold_limit=0,
            upper_threshold_limit=1,
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def __str__(self):
        return "recall"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        self._sampling_error_components = recall_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
        )

    def _calculate(self, data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(data.columns))

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return recall_score(y_true, y_pred)

    def _sampling_error(self, data: pd.DataFrame):
        return recall_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='specificity', use_case=ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationSpecificity(Metric):
    """Specificity metric."""

    def __init__(self, y_true: str, y_pred: str, y_pred_proba: Optional[str] = None):
        """Creates a new F1 instance."""
        super().__init__(
            display_name='Specificity',
            column_name='specificity',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            lower_threshold_limit=0,
            upper_threshold_limit=1,
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def __str__(self):
        return "specificity"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        self._sampling_error_components = specificity_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
        )

    def _calculate(self, data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(data.columns))

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

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

    def _sampling_error(self, data: pd.DataFrame):
        return specificity_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='accuracy', use_case=ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationAccuracy(Metric):
    """Accuracy metric."""

    def __init__(self, y_true: str, y_pred: str, y_pred_proba: Optional[str] = None):
        """Creates a new Accuracy instance."""
        super().__init__(
            display_name='Accuracy',
            column_name='accuracy',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            lower_threshold_limit=0,
            upper_threshold_limit=1,
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def __str__(self):
        return "accuracy"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        self._sampling_error_components = accuracy_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
        )

    def _calculate(self, data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(data.columns))

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

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

    def _sampling_error(self, data: pd.DataFrame):
        return accuracy_sampling_error(self._sampling_error_components, data)
