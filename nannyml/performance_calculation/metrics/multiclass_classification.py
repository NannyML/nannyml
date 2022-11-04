#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  #
#  License: Apache Software License 2.0

#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing metric utilities and implementations."""
from typing import Dict, List, Optional, Tuple, Union  # noqa: TYP001

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelBinarizer, label_binarize

from nannyml._typing import ProblemType, class_labels, model_output_column_names
from nannyml.base import _list_missing
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_calculation.metrics.base import Metric, MetricFactory
from nannyml.sampling_error.multiclass_classification import (
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


@MetricFactory.register(metric='roc_auc', use_case=ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationAUROC(Metric):
    """Area under Receiver Operating Curve metric."""

    def __init__(self, y_true: str, y_pred: str, y_pred_proba: Optional[Union[str, Dict[str, str]]] = None):
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
        self._sampling_error_components: List[Tuple] = []

    def __str__(self):
        return "roc_auc"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))

        # sampling error
        classes = class_labels(self.y_pred_proba)  # type: ignore
        binarized_y_true = list(label_binarize(reference_data[self.y_true], classes=classes).T)
        y_pred_proba = [reference_data[self.y_pred_proba[clazz]].T for clazz in classes]  # type: ignore

        self._sampling_error_components = auroc_sampling_error_components(
            y_true_reference=binarized_y_true, y_pred_proba_reference=y_pred_proba
        )

    def _calculate(self, data: pd.DataFrame):
        if not isinstance(self.y_pred_proba, Dict):
            raise InvalidArgumentsException(
                f"'y_pred_proba' is of type {type(self.y_pred_proba)}\n"
                f"multiclass use cases require 'y_pred_proba' to "
                "be a dictionary mapping classes to columns."
            )

        _list_missing([self.y_true] + model_output_column_names(self.y_pred_proba), data)

        labels, class_probability_columns = [], []
        for label in sorted(list(self.y_pred_proba.keys())):
            labels.append(label)
            class_probability_columns.append(self.y_pred_proba[label])

        y_true = data[self.y_true]
        y_pred = data[class_probability_columns]

        if y_pred.isna().all().any():
            raise InvalidArgumentsException(
                f"could not calculate metric {self.display_name}: " "prediction column contains no data"
            )

        if y_true.nunique() <= 1:
            return np.nan
        else:
            return roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro', labels=labels)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return auroc_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='f1', use_case=ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationF1(Metric):
    """F1 score metric."""

    def __init__(self, y_true: str, y_pred: str, y_pred_proba: Optional[Union[str, Dict[str, str]]] = None):
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
        self._sampling_error_components: List[Tuple] = []

    def __str__(self):
        return "f1"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], reference_data)

        # sampling error
        label_binarizer = LabelBinarizer()
        binarized_y_true = list(label_binarizer.fit_transform(reference_data[self.y_true]).T)
        binarized_y_pred = list(label_binarizer.transform(reference_data[self.y_pred]).T)

        self._sampling_error_components = f1_sampling_error_components(
            y_true_reference=binarized_y_true, y_pred_reference=binarized_y_pred
        )

    def _calculate(self, data: pd.DataFrame):
        if not isinstance(self.y_pred_proba, Dict):
            raise InvalidArgumentsException(
                f"'y_pred_proba' is of type {type(self.y_pred_proba)}\n"
                f"multiclass use cases require 'y_pred_proba' to "
                "be a dictionary mapping classes to columns."
            )

        _list_missing([self.y_true, self.y_pred], data)

        labels = sorted(list(self.y_pred_proba.keys()))
        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_pred.isna().all().any():
            raise InvalidArgumentsException(
                f"could not calculate metric {self.display_name}: " "prediction column contains no data"
            )

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return f1_score(y_true, y_pred, average='macro', labels=labels)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return f1_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='precision', use_case=ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationPrecision(Metric):
    """Precision metric."""

    def __init__(self, y_true: str, y_pred: str, y_pred_proba: Optional[Union[str, Dict[str, str]]] = None):
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
        self._sampling_error_components: List[Tuple] = []

    def __str__(self):
        return "precision"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], reference_data)

        # sampling error
        label_binarizer = LabelBinarizer()
        binarized_y_true = list(label_binarizer.fit_transform(reference_data[self.y_true]).T)
        binarized_y_pred = list(label_binarizer.transform(reference_data[self.y_pred]).T)

        self._sampling_error_components = precision_sampling_error_components(
            y_true_reference=binarized_y_true, y_pred_reference=binarized_y_pred
        )

    def _calculate(self, data: pd.DataFrame):
        if not isinstance(self.y_pred_proba, Dict):
            raise InvalidArgumentsException(
                f"'y_pred_proba' is of type {type(self.y_pred_proba)}\n"
                f"multiclass use cases require 'y_pred_proba' to "
                "be a dictionary mapping classes to columns."
            )

        _list_missing([self.y_true, self.y_pred], data)

        labels = sorted(list(self.y_pred_proba.keys()))
        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_pred.isna().all().any():
            raise InvalidArgumentsException(
                f"could not calculate metric {self.display_name}: " "prediction column contains no data"
            )

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return precision_score(y_true, y_pred, average='macro', labels=labels)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return precision_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='recall', use_case=ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationRecall(Metric):
    """Recall metric, also known as 'sensitivity'."""

    def __init__(self, y_true: str, y_pred: str, y_pred_proba: Optional[Union[str, Dict[str, str]]] = None):
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
        self._sampling_error_components: List[Tuple] = []

    def __str__(self):
        return "recall"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], reference_data)

        # sampling error
        label_binarizer = LabelBinarizer()
        binarized_y_true = list(label_binarizer.fit_transform(reference_data[self.y_true]).T)
        binarized_y_pred = list(label_binarizer.transform(reference_data[self.y_pred]).T)

        self._sampling_error_components = recall_sampling_error_components(
            y_true_reference=binarized_y_true, y_pred_reference=binarized_y_pred
        )

    def _calculate(self, data: pd.DataFrame):
        if not isinstance(self.y_pred_proba, Dict):
            raise InvalidArgumentsException(
                f"'y_pred_proba' is of type {type(self.y_pred_proba)}\n"
                f"multiclass use cases require 'y_pred_proba' to "
                "be a dictionary mapping classes to columns."
            )

        _list_missing([self.y_true, self.y_pred], data)

        labels = sorted(list(self.y_pred_proba.keys()))
        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_pred.isna().all().any():
            raise InvalidArgumentsException(
                f"could not calculate metric {self.display_name}: " "prediction column contains no data"
            )

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return recall_score(y_true, y_pred, average='macro', labels=labels)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return recall_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='specificity', use_case=ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationSpecificity(Metric):
    """Specificity metric."""

    def __init__(self, y_true: str, y_pred: str, y_pred_proba: Optional[Union[str, Dict[str, str]]] = None):
        """Creates a new Specificity instance."""
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
        self._sampling_error_components: List[Tuple] = []

    def __str__(self):
        return "specificity"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], reference_data)

        # sampling error
        label_binarizer = LabelBinarizer()
        binarized_y_true = list(label_binarizer.fit_transform(reference_data[self.y_true]).T)
        binarized_y_pred = list(label_binarizer.transform(reference_data[self.y_pred]).T)

        self._sampling_error_components = specificity_sampling_error_components(
            y_true_reference=binarized_y_true, y_pred_reference=binarized_y_pred
        )

    def _calculate(self, data: pd.DataFrame):
        if not isinstance(self.y_pred_proba, Dict):
            raise InvalidArgumentsException(
                f"'y_pred_proba' is of type {type(self.y_pred_proba)}\n"
                f"multiclass use cases require 'y_pred_proba' to "
                "be a dictionary mapping classes to columns."
            )

        _list_missing([self.y_true, self.y_pred], data)

        labels = sorted(list(self.y_pred_proba.keys()))
        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_pred.isna().all().any():
            raise InvalidArgumentsException(
                f"could not calculate metric {self.display_name}: prediction column contains no data"
            )

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            MCM = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
            tn_sum = MCM[:, 0, 0]
            fp_sum = MCM[:, 0, 1]
            class_wise_specificity = tn_sum / (tn_sum + fp_sum)
            return np.mean(class_wise_specificity)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return specificity_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='accuracy', use_case=ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationAccuracy(Metric):
    """Accuracy metric."""

    def __init__(self, y_true: str, y_pred: str, y_pred_proba: Optional[Union[str, Dict[str, str]]] = None):
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
        _list_missing([self.y_true, self.y_pred], reference_data)

        # sampling error
        label_binarizer = LabelBinarizer()
        binarized_y_true = label_binarizer.fit_transform(reference_data[self.y_true])
        binarized_y_pred = label_binarizer.transform(reference_data[self.y_pred])

        self._sampling_error_components = accuracy_sampling_error_components(
            y_true_reference=binarized_y_true, y_pred_reference=binarized_y_pred
        )

    def _calculate(self, data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], data)

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_pred.isna().all().any():
            raise InvalidArgumentsException(
                f"could not calculate metric '{self.display_name}': " "prediction column contains no data"
            )

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return accuracy_score(y_true, y_pred)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return accuracy_sampling_error(self._sampling_error_components, data)
