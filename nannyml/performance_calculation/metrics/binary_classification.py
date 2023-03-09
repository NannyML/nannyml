#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from typing import Optional, Tuple, Union, List

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
    business_value_sampling_error,
    business_value_sampling_error_components,
)


@MetricFactory.register(metric='roc_auc', use_case=ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationAUROC(Metric):
    """Area under Receiver Operating Curve metric."""

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        y_pred_proba: Optional[str] = None,
        **kwargs,
    ):
        """Creates a new AUROC instance."""
        super().__init__(
            name='roc_auc',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            lower_threshold_limit=0,
            upper_threshold_limit=1,
            components=[('ROC AUC', 'roc_auc')],
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

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        y_pred_proba: Optional[str] = None,
        **kwargs,
    ):
        """Creates a new F1 instance."""
        super().__init__(
            name='f1',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            lower_threshold_limit=0,
            upper_threshold_limit=1,
            components=[('F1', 'f1')],
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

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        y_pred_proba: Optional[str] = None,
        **kwargs,
    ):
        """Creates a new Precision instance."""
        super().__init__(
            name='precision',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            lower_threshold_limit=0,
            upper_threshold_limit=1,
            components=[('Precision', 'precision')],
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

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        y_pred_proba: Optional[str] = None,
        **kwargs,
    ):
        """Creates a new Recall instance."""
        super().__init__(
            name='recall',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            lower_threshold_limit=0,
            upper_threshold_limit=1,
            components=[('Recall', 'recall')],
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

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        y_pred_proba: Optional[str] = None,
        **kwargs,
    ):
        """Creates a new F1 instance."""
        super().__init__(
            name='specificity',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            lower_threshold_limit=0,
            upper_threshold_limit=1,
            components=[('Specificity', 'specificity')],
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

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        y_pred_proba: Optional[str] = None,
        **kwargs,
    ):
        """Creates a new Accuracy instance."""
        super().__init__(
            name='accuracy',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            lower_threshold_limit=0,
            upper_threshold_limit=1,
            components=[('Accuracy', 'accuracy')],
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


@MetricFactory.register(metric='business_value', use_case=ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationBusinessValue(Metric):
    """Accuracy metric."""

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        business_value_matrix: Union[List, np.ndarray],
        normalize_business_value: Optional[str] = None,
        y_pred_proba: Optional[str] = None,
        **kwargs,
    ):
        """Creates a new Accuracy instance."""

        if normalize_business_value not in [None, "per_prediction"]:
            raise InvalidArgumentsException(
                f"normalize_business_value must be None or 'per_prediction', but got {normalize_business_value}"
            )

        super().__init__(
            name='business_value',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            lower_threshold_limit=-np.inf,
            upper_threshold_limit=np.inf,
            components=[('Business Value', 'business_value')],
        )

        if business_value_matrix is None:
            raise ValueError("business_value_matrix must be provided for 'business_value' metric")

        if not (isinstance(business_value_matrix, np.ndarray) or isinstance(business_value_matrix, list)):
            raise ValueError(
                f"business_value_matrix must be a numpy array or a list, but got {type(business_value_matrix)}"
            )

        if isinstance(business_value_matrix, list):
            business_value_matrix = np.array(business_value_matrix)

        if business_value_matrix.shape != (2, 2):
            raise ValueError(
                f"business_value_matrix must have shape (2,2), but got matrix of shape {business_value_matrix.shape}"
            )

        self.business_value_matrix = business_value_matrix
        self.normalize_business_value: Optional[str] = normalize_business_value

        # sampling error
        self._sampling_error_components: Tuple = ()

    def __str__(self):
        return "business_value"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        self._sampling_error_components = business_value_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
            business_value_matrix=self.business_value_matrix,
            normalize_business_value=self.normalize_business_value,
        )

    def _calculate(self, data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(data.columns))

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_pred.isna().all():
            raise InvalidArgumentsException(
                f"could not calculate metric '{self.name}': " "prediction column contains no data"
            )

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        tp_value = self.business_value_matrix[1, 1]
        tn_value = self.business_value_matrix[0, 0]
        fp_value = self.business_value_matrix[0, 1]
        fn_value = self.business_value_matrix[1, 0]

        num_tp = num_tp = np.sum(np.logical_and(y_pred, y_true))
        num_tn = np.sum(np.logical_and(np.logical_not(y_pred), np.logical_not(y_true)))
        num_fp = np.sum(np.logical_and(y_pred, np.logical_not(y_true)))
        num_fn = np.sum(np.logical_and(np.logical_not(y_pred), y_true))

        business_value = num_tp * tp_value + num_tn * tn_value + num_fp * fp_value + num_fn * fn_value

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            if self.normalize_business_value is None:
                return business_value
            else:  # normalize must be 'per_prediction'
                return business_value / len(y_true)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return business_value_sampling_error(self._sampling_error_components, data)
