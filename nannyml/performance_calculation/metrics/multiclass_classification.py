#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  #
#  License: Apache Software License 2.0

"""Module containing metric utilities and implementations."""
import warnings
from typing import Dict, List, Optional, Tuple, Union  # noqa: TYP001

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import LabelBinarizer, label_binarize

from nannyml._typing import ProblemType, class_labels
from nannyml.base import _list_missing, common_nan_removal
from nannyml.chunk import Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_calculation.metrics.base import Metric, MetricFactory
from nannyml.sampling_error.multiclass_classification import (
    accuracy_sampling_error,
    accuracy_sampling_error_components,
    auroc_sampling_error,
    auroc_sampling_error_components,
    f1_sampling_error,
    f1_sampling_error_components,
    multiclass_confusion_matrix_sampling_error,
    multiclass_confusion_matrix_sampling_error_components,
    precision_sampling_error,
    precision_sampling_error_components,
    recall_sampling_error,
    recall_sampling_error_components,
    specificity_sampling_error,
    specificity_sampling_error_components,
    average_precision_sampling_error_components,
    average_precision_sampling_error,
    business_value_sampling_error_components,
    business_value_sampling_error,
)
from nannyml.thresholds import Threshold, calculate_threshold_values


@MetricFactory.register(metric='roc_auc', use_case=ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationAUROC(Metric):
    """Area under Receiver Operating Curve metric."""

    y_pred_proba: Dict[str, str]

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        threshold: Threshold,
        y_pred_proba: Dict[str, str],
        **kwargs,
    ):
        """Creates a new AUROC instance.

        Parameters
        ----------
        y_true: str
            The name of the column containing target values.
        y_pred: str
            The name of the column containing your model predictions.
        threshold: Threshold
            The Threshold instance that determines how the lower and upper threshold values will be calculated.
        y_pred_proba: Union[str, Dict[str, str]]
            Name(s) of the column(s) containing your model output.

                - For binary classification, pass a single string refering to the model output column.
                - For multiclass classification, pass a dictionary that maps a class string to the column name \
                containing model outputs for that class.
        """
        super().__init__(
            name='roc_auc',
            y_true=y_true,
            y_pred=y_pred,
            threshold=threshold,
            y_pred_proba=y_pred_proba,
            lower_threshold_limit=0,
            upper_threshold_limit=1,
            components=[("ROC AUC", "roc_auc")],
        )
        self.y_pred_proba: Dict[str, str]
        self.classes: List[str] = [""]
        self.class_probability_columns: List[str]
        self._sampling_error_components: List[Tuple] = []

    def __str__(self):
        """Get string representation of metric."""
        return "roc_auc"

    def _fit(self, reference_data: pd.DataFrame):
        # set up sorted classes and prob_column_names to use across metric class
        self.classes = class_labels(self.y_pred_proba)
        self.class_probability_columns = [self.y_pred_proba[clazz] for clazz in self.classes]

        _list_missing([self.y_true] + self.class_probability_columns, list(reference_data.columns))
        reference_data, empty = common_nan_removal(
            reference_data[[self.y_true] + self.class_probability_columns],
            [self.y_true] + self.class_probability_columns,
        )
        if empty:
            self._sampling_error_components = [(np.nan, 0) for clasz in self.classes]
            # TODO: Ideally we would also raise an error here!
        else:
            # test if reference data are represented correctly
            observed_classes = set(reference_data[self.y_true].unique())
            if not observed_classes == set(self.classes):
                self._logger.error(
                    "The specified classification classes are not the same as the classes observed in the reference"
                    "targets."
                )
                raise InvalidArgumentsException(
                    "y_pred_proba class and class probabilities dictionary does not match reference data."
                )

            # sampling error
            binarized_y_true = list(label_binarize(reference_data[self.y_true], classes=self.classes).T)
            y_pred_proba = [reference_data[self.y_pred_proba[clazz]].T for clazz in self.classes]
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

        _list_missing([self.y_true] + self.class_probability_columns, data)
        data, empty = common_nan_removal(
            data[[self.y_true] + self.class_probability_columns], [self.y_true] + self.class_probability_columns
        )
        if empty:
            _message = f"Too many missing values, cannot calculate {self.display_name}. " f"Returning NaN."
            self._logger.warning(_message)
            warnings.warn(_message)
            return np.nan

        y_true = data[self.y_true]
        y_pred_proba = data[self.class_probability_columns]

        if set(y_true.unique()) != set(self.classes):
            _message = (
                f"'{self.y_true}' does not contain all reported classes, cannot calculate {self.display_name}. "
                "Returning NaN."
            )
            warnings.warn(_message)
            self._logger.warning(_message)
            return np.nan
        else:
            return roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro', labels=self.classes)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        _list_missing([self.y_true] + self.class_probability_columns, data)
        data, empty = common_nan_removal(
            data[[self.y_true] + self.class_probability_columns], [self.y_true] + self.class_probability_columns
        )
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. " f"Returning NaN."
            )
            return np.nan
        else:
            return auroc_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='f1', use_case=ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationF1(Metric):
    """F1 score metric."""

    y_pred: str
    y_pred_proba: Dict[str, str]

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        threshold: Threshold,
        y_pred_proba: Optional[Union[str, Dict[str, str]]] = None,
        **kwargs,
    ):
        """Creates a new F1 instance.

        Parameters
        ----------
        y_true: str
            The name of the column containing target values.
        y_pred: str
            The name of the column containing your model predictions.
        threshold: Threshold
            The Threshold instance that determines how the lower and upper threshold values will be calculated.
        y_pred_proba: Union[str, Dict[str, str]]
            Name(s) of the column(s) containing your model output.

                - For binary classification, pass a single string refering to the model output column.
                - For multiclass classification, pass a dictionary that maps a class string to the column name \
                containing model outputs for that class.
        """
        super().__init__(
            name='f1',
            y_true=y_true,
            y_pred=y_pred,
            threshold=threshold,
            y_pred_proba=y_pred_proba,
            lower_threshold_limit=0,
            upper_threshold_limit=1,
            components=[("F1", "f1")],
        )

        # sampling error
        self._sampling_error_components: List[Tuple] = []

    def __str__(self):
        """Get string representation of metric."""
        return "f1"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], reference_data)
        classes = class_labels(self.y_pred_proba)
        reference_data, empty = common_nan_removal(
            reference_data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            self._sampling_error_components = [(np.nan, 0) for clazz in classes]
        else:
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
        data, empty = common_nan_removal(data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred])
        if empty:
            warnings.warn(f"Too many missing values, cannot calculate {self.display_name}. " f"Returning NaN.")
            return np.nan

        labels = sorted(list(self.y_pred_proba.keys()))
        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_true.nunique() <= 1:
            warnings.warn(
                f"'{self.y_true}' only contains a single class, cannot calculate {self.display_name}. Returning NaN."
            )
            return np.nan
        elif y_pred.nunique() <= 1:
            warnings.warn(
                f"'{self.y_pred}' only contains a single class, cannot calculate {self.display_name}. Returning NaN."
            )
            return np.nan
        else:
            return f1_score(y_true, y_pred, average='macro', labels=labels)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        _list_missing([self.y_true, self.y_pred], data)
        data, empty = common_nan_removal(data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred])
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. " "Returning NaN."
            )
            return np.nan
        else:
            return f1_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='precision', use_case=ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationPrecision(Metric):
    """Precision metric."""

    y_pred: str
    y_pred_proba: Dict[str, str]

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        threshold: Threshold,
        y_pred_proba: Optional[Union[str, Dict[str, str]]] = None,
        **kwargs,
    ):
        """Creates a new Precision instance.

        Parameters
        ----------
        y_true: str
            The name of the column containing target values.
        y_pred: str
            The name of the column containing your model predictions.
        threshold: Threshold
            The Threshold instance that determines how the lower and upper threshold values will be calculated.
        y_pred_proba: Union[str, Dict[str, str]]
            Name(s) of the column(s) containing your model output.

                - For binary classification, pass a single string refering to the model output column.
                - For multiclass classification, pass a dictionary that maps a class string to the column name \
                containing model outputs for that class.
        """
        super().__init__(
            name='precision',
            y_true=y_true,
            y_pred=y_pred,
            threshold=threshold,
            y_pred_proba=y_pred_proba,
            lower_threshold_limit=0,
            upper_threshold_limit=1,
            components=[("Precision", "precision")],
        )

        # sampling error
        self._sampling_error_components: List[Tuple] = []

    def __str__(self):
        """Get string representation of metric."""
        return "precision"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], reference_data)
        classes = class_labels(self.y_pred_proba)
        reference_data, empty = common_nan_removal(
            reference_data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            self._sampling_error_components = [(np.nan, 0) for clazz in classes]
        else:
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
        data, empty = common_nan_removal(data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred])
        if empty:
            warnings.warn(f"Too many missing values, cannot calculate {self.display_name}. " f"Returning NaN.")
            return np.nan

        labels = sorted(list(self.y_pred_proba.keys()))
        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_true.nunique() <= 1:
            warnings.warn(
                f"'{self.y_true}' only contains a single class, cannot calculate {self.display_name}. Returning NaN."
            )
            return np.nan
        elif y_pred.nunique() <= 1:
            warnings.warn(
                f"'{self.y_pred}' only contains a single class, cannot calculate {self.display_name}. Returning NaN."
            )
            return np.nan
        else:
            return precision_score(y_true, y_pred, average='macro', labels=labels)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        _list_missing([self.y_true, self.y_pred], data)
        data, empty = common_nan_removal(data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred])
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. " "Returning NaN."
            )
            return np.nan
        else:
            return precision_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='recall', use_case=ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationRecall(Metric):
    """Recall metric, also known as 'sensitivity'."""

    y_pred: str
    y_pred_proba: Dict[str, str]

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        threshold: Threshold,
        y_pred_proba: Optional[Union[str, Dict[str, str]]] = None,
        **kwargs,
    ):
        """Creates a new Recall instance.

        Parameters
        ----------
        y_true: str
            The name of the column containing target values.
        y_pred: str
            The name of the column containing your model predictions.
        threshold: Threshold
            The Threshold instance that determines how the lower and upper threshold values will be calculated.
        y_pred_proba: Union[str, Dict[str, str]]
            Name(s) of the column(s) containing your model output.

                - For binary classification, pass a single string refering to the model output column.
                - For multiclass classification, pass a dictionary that maps a class string to the column name \
                containing model outputs for that class.
        """
        super().__init__(
            name='recall',
            y_true=y_true,
            y_pred=y_pred,
            threshold=threshold,
            y_pred_proba=y_pred_proba,
            lower_threshold_limit=0,
            upper_threshold_limit=1,
            components=[("Recall", "recall")],
        )

        # sampling error
        self._sampling_error_components: List[Tuple] = []

    def __str__(self):
        """Get string representation of metric."""
        return "recall"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], reference_data)
        classes = class_labels(self.y_pred_proba)
        reference_data, empty = common_nan_removal(
            reference_data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            self._sampling_error_components = [(np.nan, 0) for clazz in classes]
        else:
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
        data, empty = common_nan_removal(data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred])
        if empty:
            warnings.warn(f"Too many missing values, cannot calculate {self.display_name}. " f"Returning NaN.")
            return np.nan

        labels = sorted(list(self.y_pred_proba.keys()))
        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_true.nunique() <= 1:
            warnings.warn(
                f"'{self.y_true}' only contains a single class, cannot calculate {self.display_name}. Returning NaN."
            )
            return np.nan
        elif y_pred.nunique() <= 1:
            warnings.warn(
                f"'{self.y_pred}' only contains a single class, cannot calculate {self.display_name}. Returning NaN."
            )
            return np.nan
        else:
            return recall_score(y_true, y_pred, average='macro', labels=labels)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        _list_missing([self.y_true, self.y_pred], data)
        data, empty = common_nan_removal(data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred])
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. " "Returning NaN."
            )
            return np.nan
        else:
            return recall_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='specificity', use_case=ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationSpecificity(Metric):
    """Specificity metric."""

    y_pred: str
    y_pred_proba: Dict[str, str]

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        threshold: Threshold,
        y_pred_proba: Optional[Union[str, Dict[str, str]]] = None,
        **kwargs,
    ):
        """Creates a new Specificity instance.

        Parameters
        ----------
        y_true: str
            The name of the column containing target values.
        y_pred: str
            The name of the column containing your model predictions.
        threshold: Threshold
            The Threshold instance that determines how the lower and upper threshold values will be calculated.
        y_pred_proba: Union[str, Dict[str, str]]
            Name(s) of the column(s) containing your model output.

                - For binary classification, pass a single string refering to the model output column.
                - For multiclass classification, pass a dictionary that maps a class string to the column name \
                containing model outputs for that class.
        """
        super().__init__(
            name='specificity',
            y_true=y_true,
            y_pred=y_pred,
            threshold=threshold,
            y_pred_proba=y_pred_proba,
            lower_threshold_limit=0,
            upper_threshold_limit=1,
            components=[("Specificity", "specificity")],
        )

        # sampling error
        self._sampling_error_components: List[Tuple] = []

    def __str__(self):
        """Get string representation of metric."""
        return "specificity"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], reference_data)
        classes = class_labels(self.y_pred_proba)
        reference_data, empty = common_nan_removal(
            reference_data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            self._sampling_error_components = [(np.nan, 0) for clazz in classes]
        else:
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
        data, empty = common_nan_removal(data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred])
        if empty:
            warnings.warn(f"Too many missing values, cannot calculate {self.display_name}. " f"Returning NaN.")
            return np.nan

        labels = sorted(list(self.y_pred_proba.keys()))
        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_true.nunique() <= 1:
            warnings.warn(
                f"'{self.y_true}' only contains a single class, cannot calculate {self.display_name}. Returning NaN."
            )
            return np.nan
        elif y_pred.nunique() <= 1:
            warnings.warn(
                f"'{self.y_pred}' only contains a single class, cannot calculate {self.display_name}. Returning NaN."
            )
            return np.nan
        else:
            MCM = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
            tn_sum = MCM[:, 0, 0]
            fp_sum = MCM[:, 0, 1]
            class_wise_specificity = tn_sum / (tn_sum + fp_sum)
            return np.mean(class_wise_specificity)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        _list_missing([self.y_true, self.y_pred], data)
        data, empty = common_nan_removal(data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred])
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. " "Returning NaN."
            )
            return np.nan
        else:
            return specificity_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='accuracy', use_case=ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationAccuracy(Metric):
    """Accuracy metric."""

    y_pred: str
    y_pred_proba: Dict[str, str]

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        threshold: Threshold,
        y_pred_proba: Optional[Union[str, Dict[str, str]]] = None,
        **kwargs,
    ):
        """Creates a new Accuracy instance.

        Parameters
        ----------
        y_true: str
            The name of the column containing target values.
        y_pred: str
            The name of the column containing your model predictions.
        threshold: Threshold
            The Threshold instance that determines how the lower and upper threshold values will be calculated.
        y_pred_proba: Union[str, Dict[str, str]]
            Name(s) of the column(s) containing your model output.

                - For binary classification, pass a single string refering to the model output column.
                - For multiclass classification, pass a dictionary that maps a class string to the column name \
                containing model outputs for that class.
        """
        super().__init__(
            name='accuracy',
            y_true=y_true,
            y_pred=y_pred,
            threshold=threshold,
            y_pred_proba=y_pred_proba,
            lower_threshold_limit=0,
            upper_threshold_limit=1,
            components=[("Accuracy", "accuracy")],
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def __str__(self):
        """Get string representation of metric."""
        return "accuracy"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], reference_data)
        reference_data, empty = common_nan_removal(
            reference_data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            self._sampling_error_components = (np.nan,)
        else:
            # sampling error
            label_binarizer = LabelBinarizer()
            binarized_y_true = label_binarizer.fit_transform(reference_data[self.y_true])
            binarized_y_pred = label_binarizer.transform(reference_data[self.y_pred])

            self._sampling_error_components = accuracy_sampling_error_components(
                y_true_reference=binarized_y_true, y_pred_reference=binarized_y_pred
            )

    def _calculate(self, data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], data)
        data, empty = common_nan_removal(data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred])
        if empty:
            warnings.warn(f"Too many missing values, cannot calculate {self.display_name}. " f"Returning NaN.")
            return np.nan

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        return accuracy_score(y_true, y_pred)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        _list_missing([self.y_true, self.y_pred], data)
        data, empty = common_nan_removal(data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred])
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. " "Returning NaN."
            )
            return np.nan
        else:
            return accuracy_sampling_error(self._sampling_error_components, data)


@MetricFactory.register('confusion_matrix', ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationConfusionMatrix(Metric):
    """Multiclass Confusion Matrix metric."""

    y_pred: str
    y_pred_proba: Dict[str, str]
    # classes: List[str]

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        threshold: Threshold,
        y_pred_proba: Optional[Union[str, Dict[str, str]]] = None,
        normalize_confusion_matrix: Optional[str] = None,
        **kwargs,
    ):
        """Creates a new confusion matrix instance."""
        super().__init__(
            name='confusion_matrix',
            y_true=y_true,
            y_pred=y_pred,
            threshold=threshold,
            y_pred_proba=y_pred_proba,
            components=[("None", "none")],
            lower_threshold_limit=0,
        )

        self.normalize_confusion_matrix: Optional[str] = normalize_confusion_matrix
        self.upper_threshold_value_limit: Optional[float] = 1.0 if normalize_confusion_matrix else None

        self.classes: Optional[List[str]] = None

    def __str__(self):
        """Get string representation of metric."""
        return "confusion_matrix"

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
        # _fit
        # realized perf on chunks
        # set thresholds

        self._fit(reference_data)

        reference_chunks = chunker.split(reference_data)
        reference_chunk_results = np.asarray([self._calculate(chunk.data) for chunk in reference_chunks])

        self.alert_thresholds = self._multiclass_confusion_matrix_alert_thresholds(
            reference_chunk_results=reference_chunk_results,
        )

    def _multiclass_confusion_matrix_alert_thresholds(
        self,
        reference_chunk_results: np.ndarray,
    ) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """Calculate the alert thresholds for the confusion matrix.

        Args:
            reference_chunk_results: The confusion matrix for each chunk of the reference data.

        Returns:
            The alert thresholds for the confusion matrix.
        """
        alert_thresholds = {}

        if self.classes is None:
            raise ValueError("classes must be set before calling this method")

        num_classes = len(self.classes)

        for i in range(num_classes):
            for j in range(num_classes):
                lower_threshold_value, upper_threshold_value = calculate_threshold_values(
                    threshold=self.threshold,
                    data=reference_chunk_results[:, i, j],
                    lower_threshold_value_limit=self.lower_threshold_value_limit,
                    upper_threshold_value_limit=self.upper_threshold_value_limit,
                )
                alert_thresholds[f'true_{self.classes[i]}_pred_{self.classes[j]}'] = (
                    lower_threshold_value,
                    upper_threshold_value,
                )

        return alert_thresholds

    def _fit(self, reference_data: pd.DataFrame):
        self.classes = sorted(reference_data[self.y_true].unique())
        self.components = self._get_components(self.classes)

        _list_missing([self.y_true, self.y_pred], reference_data)
        reference_data, empty = common_nan_removal(
            reference_data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            self._sampling_error_components = np.full((len(self.classes), len(self.classes)), np.nan), 0
        else:
            # sampling error
            self.sampling_error_components = multiclass_confusion_matrix_sampling_error_components(
                y_true_reference=reference_data[self.y_true],
                y_pred_reference=reference_data[self.y_pred],
                normalize_confusion_matrix=self.normalize_confusion_matrix,
            )

    def _get_components(self, classes: List[str]) -> List[Tuple[str, str]]:
        components = []

        for true_class in classes:
            for pred_class in classes:
                components.append(
                    (
                        f"true class: '{true_class}', predicted class: '{pred_class}'",
                        f'true_{true_class}_pred_{pred_class}',
                    )
                )

        return components

    def _calculate(self, data: pd.DataFrame) -> Union[np.ndarray, float]:
        _list_missing([self.y_true, self.y_pred], data)
        data, empty = common_nan_removal(data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred])
        if empty:
            warnings.warn(f"Too many missing values, cannot calculate {self.display_name}. " f"Returning NaN.")
            return np.nan

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            cm = confusion_matrix(y_true, y_pred, labels=self.classes, normalize=self.normalize_confusion_matrix)
            return cm

    def sampling_error(self, data: pd.DataFrame):
        """Calculates the sampling error with respect to the reference data for a given chunk of data.

        Parameters
        ----------
        data: pd.DataFrame
            The data to calculate the sampling error on, with respect to the reference data.

        Returns
        -------
        sampling_error: float
            The expected sampling error.

        """
        _list_missing([self.y_true, self.y_pred], data)
        data, empty = common_nan_removal(data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred])
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. " "Returning NaN."
            )
            num_classes: int = len(self.classes)  # type: ignore
            return np.full((num_classes, num_classes), np.nan)
        else:
            return multiclass_confusion_matrix_sampling_error(self.sampling_error_components, data)

    def get_chunk_record(self, chunk_data: pd.DataFrame) -> Dict[str, Union[float, bool]]:
        """Create results for provided chunk data."""
        if self.classes is None:
            raise ValueError("classes must be set before calling this method")

        realized_cm = self._calculate(chunk_data)
        sampling_errors = self.sampling_error(chunk_data)

        if isinstance(realized_cm, float):
            realized_cm = np.full((len(self.classes), len(self.classes)), np.nan)

        chunk_record = {}

        for true_class in self.classes:
            for pred_class in self.classes:
                column_name = f'true_{true_class}_pred_{pred_class}'

                chunk_record[f"{column_name}_sampling_error"] = sampling_errors[
                    self.classes.index(true_class), self.classes.index(pred_class)
                ]

                chunk_record[f"{column_name}"] = realized_cm[
                    self.classes.index(true_class), self.classes.index(pred_class)
                ]

                lower_threshold, upper_threshold = self.alert_thresholds[f"true_{true_class}_pred_{pred_class}"]
                chunk_record[f"{column_name}_upper_threshold"] = upper_threshold
                chunk_record[f"{column_name}_lower_threshold"] = lower_threshold

                chunk_record[f"{column_name}_alert"] = (
                    self.alert_thresholds is not None and (chunk_record[f"{column_name}"] < lower_threshold)
                ) or (self.alert_thresholds is not None and (chunk_record[f"{column_name}"] > upper_threshold))

        return chunk_record


@MetricFactory.register(metric='average_precision', use_case=ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationAP(Metric):
    """Average Precision metric."""

    y_pred_proba: Dict[str, str]

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        threshold: Threshold,
        y_pred_proba: Dict[str, str],
        **kwargs,
    ):
        """Creates a new AP instance.

        Parameters
        ----------
        y_true: str
            The name of the column containing target values.
        y_pred: str
            The name of the column containing your model predictions.
        threshold: Threshold
            The Threshold instance that determines how the lower and upper threshold values will be calculated.
        y_pred_proba: Union[str, Dict[str, str]]
            Name(s) of the column(s) containing your model output.

                - For binary classification, pass a single string refering to the model output column.
                - For multiclass classification, pass a dictionary that maps a class string to the column name \
                containing model outputs for that class.
        """
        super().__init__(
            name='average_precision',
            y_true=y_true,
            y_pred=y_pred,
            threshold=threshold,
            y_pred_proba=y_pred_proba,
            lower_threshold_limit=0,
            upper_threshold_limit=1,
            components=[("Average Precision", "average_precision")],
        )
        # FIXME: Should we check the y_pred_proba argument here to ensure it's a dict?
        self.y_pred_proba: Dict[str, str]

        # sampling error
        self._sampling_error_components: List[Tuple] = []

        # classes and class probability columns
        self.classes: List[str] = [""]
        self.class_probability_columns: List[str]

    def __str__(self):
        """Get string representation of metric."""
        return "average_precision"

    def _fit(self, reference_data: pd.DataFrame):
        # set up sorted classes and prob_column_names to use across metric class
        self.classes = class_labels(self.y_pred_proba)
        self.class_probability_columns = [self.y_pred_proba[clazz] for clazz in self.classes]
        _list_missing([self.y_true] + self.class_probability_columns, list(reference_data.columns))
        reference_data, empty = common_nan_removal(
            reference_data[[self.y_true] + self.class_probability_columns],
            [self.y_true] + self.class_probability_columns,
        )
        if empty:
            self._sampling_error_components = [(np.nan, 0) for class_col in self.class_probability_columns]
        else:
            # sampling error
            binarized_y_true = list(label_binarize(reference_data[self.y_true], classes=self.classes).T)
            y_pred_proba = [reference_data[self.y_pred_proba[clazz]].T for clazz in self.classes]
            self._sampling_error_components = average_precision_sampling_error_components(
                y_true_reference=binarized_y_true, y_pred_proba_reference=y_pred_proba
            )

    def _calculate(self, data: pd.DataFrame):
        if not isinstance(self.y_pred_proba, Dict):
            raise InvalidArgumentsException(
                f"'y_pred_proba' is of type {type(self.y_pred_proba)}\n"
                f"multiclass use cases require 'y_pred_proba' to "
                "be a dictionary mapping classes to columns."
            )

        # class_y_pred_proba_columns = model_output_column_names(self.y_pred_proba)
        _list_missing([self.y_true] + self.class_probability_columns, data)
        data, empty = common_nan_removal(
            data[[self.y_true] + self.class_probability_columns], [self.y_true] + self.class_probability_columns
        )
        if empty:
            warnings.warn(f"Too many missing values, cannot calculate {self.display_name}. " f"Returning NaN.")
            return np.nan

        y_true = data[self.y_true]
        y_pred_proba = data[self.class_probability_columns]

        if y_true.nunique() <= 1:
            warnings.warn(
                f"'{self.y_true}' only contains a single class for chunk, cannot calculate {self.display_name}. "
                "Returning NaN."
            )
            return np.nan
        else:
            # https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
            # average_precision_score always performs OVR averaging
            return average_precision_score(y_true, y_pred_proba, average='macro')

    def _sampling_error(self, data: pd.DataFrame) -> float:
        _list_missing([self.y_true] + self.class_probability_columns, data)
        data, empty = common_nan_removal(
            data[[self.y_true] + self.class_probability_columns], [self.y_true] + self.class_probability_columns
        )
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. " f"Returning NaN."
            )
            return np.nan
        else:
            return average_precision_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='business_value', use_case=ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationBusinessValue(Metric):
    """Business Value metric."""

    y_pred: str
    y_pred_proba: Dict[str, str]

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        threshold: Threshold,
        business_value_matrix: Union[List, np.ndarray],
        normalize_business_value: Optional[str] = None,
        y_pred_proba: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """Creates a new Business Value instance.

        Parameters
        ----------
        y_true: str
            The name of the column containing target values.
        y_pred: str
            The name of the column containing your model predictions.
        threshold: Threshold
            The Threshold instance that determines how the lower and upper threshold values will be calculated.
        business_value_matrix: Union[List, np.ndarray]
            A nxn matrix that specifies the value of each cell in the confusion matrix.
            The format of the business value matrix must be specified so that each element represents the business
            value of it's respective confusion matrix element. Hence the element on the i-th row and j-column of the
            business value matrix tells us the value of the i-th target while we predicted the j-th value.
            It can be provided as a list of lists or a numpy array.
        normalize_business_value: Optional[str], default=None
            Determines how the business value will be normalized. Allowed values are None and 'per_prediction'.
        y_pred_proba: Optional[str], default=None
            Name(s) of the column(s) containing your model output. For binary classification, pass a single string
            refering to the model output column.
        """
        if normalize_business_value not in [None, "per_prediction"]:
            raise InvalidArgumentsException(
                f"normalize_business_value must be None or 'per_prediction', but got {normalize_business_value}"
            )

        super().__init__(
            name='business_value',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            threshold=threshold,
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
        _rows, _columns = business_value_matrix.shape
        if _rows != _columns:
            raise InvalidArgumentsException(
                f"business_value_matrix is not a square matrix but has shape: {(_rows, _columns)}"
            )

        self.business_value_matrix = business_value_matrix
        self.normalize_business_value: Optional[str] = normalize_business_value

        # sampling error
        self._sampling_error_components: Tuple = ()

        # if y_pred_proba is provided uses this to get information about number of classes in the problem.
        if y_pred_proba:
            if not isinstance(self.y_pred_proba, Dict):
                raise InvalidArgumentsException(
                    f"'y_pred_proba' is of type {type(self.y_pred_proba)}\n"
                    f"multiclass use cases require 'y_pred_proba' to "
                    "be a dictionary mapping classes to columns."
                )
            self.y_pred_proba: Dict[str, str] = y_pred_proba
            self.classes: List[str] = class_labels(self.y_pred_proba)

    def __str__(self):
        """Get string representation of metric."""
        return "business_value"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        data, empty = common_nan_removal(reference_data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred])
        if empty:
            self._sampling_error_components = np.nan, self.normalize_business_value
        else:
            # get class number from y_pred_proba if provided otherwise from reference y_true
            # this way the code will work even if some classes are missing from reference
            # provided the business value matrix is constructed correctly.
            if self.classes:
                num_classes = len(self.classes)
                _classes = self.classes
            else:
                num_classes = reference_data[self.y_true].nunique()
                _classes = sorted(list(reference_data[self.y_true].unique))
            if num_classes != self.business_value_matrix.shape[0]:
                raise InvalidArgumentsException(
                    f"business_value_matrix has shape {self.business_value_matrix.shape} "
                    "but we have {num_classes} classes!"
                )
            self._sampling_error_components = business_value_sampling_error_components(
                y_true_reference=data[self.y_true],
                y_pred_reference=data[self.y_pred],
                business_value_matrix=self.business_value_matrix,
                classes=_classes,
                normalize_business_value=self.normalize_business_value,
            )

    def _calculate(self, data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(data.columns))
        data, empty = common_nan_removal(data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred])
        if empty:
            warnings.warn(f"'{self.y_true}' contains no data, cannot calculate business value. Returning NaN.")
            return np.nan

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        cm = confusion_matrix(y_true, y_pred, labels=self.classes)
        if self.normalize_business_value == 'per_prediction':
            with np.errstate(all="ignore"):
                cm = cm / cm.sum(axis=0, keepdims=True)
            cm = np.nan_to_num(cm)

        return (self.business_value_matrix * cm).sum()

    def _sampling_error(self, data: pd.DataFrame) -> float:
        data, empty = common_nan_removal(data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred])
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. " "Returning NaN."
            )
            return np.nan
        else:
            return business_value_sampling_error(self._sampling_error_components, data)
