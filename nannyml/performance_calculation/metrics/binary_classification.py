#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from nannyml._typing import ProblemType
from nannyml.base import _list_missing, _remove_nans
from nannyml.chunk import Chunk, Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_calculation.metrics.base import Metric, MetricFactory
from nannyml.sampling_error.binary_classification import (
    accuracy_sampling_error,
    accuracy_sampling_error_components,
    auroc_sampling_error,
    auroc_sampling_error_components,
    business_value_sampling_error,
    business_value_sampling_error_components,
    f1_sampling_error,
    f1_sampling_error_components,
    false_negative_sampling_error,
    false_negative_sampling_error_components,
    false_positive_sampling_error,
    false_positive_sampling_error_components,
    precision_sampling_error,
    precision_sampling_error_components,
    recall_sampling_error,
    recall_sampling_error_components,
    specificity_sampling_error,
    specificity_sampling_error_components,
    true_negative_sampling_error,
    true_negative_sampling_error_components,
    true_positive_sampling_error,
    true_positive_sampling_error_components,
)
from nannyml.thresholds import Threshold, calculate_threshold_values


@MetricFactory.register(metric='roc_auc', use_case=ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationAUROC(Metric):
    """Area under Receiver Operating Curve metric."""

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        threshold: Threshold,
        y_pred_proba: Optional[str] = None,
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
        y_pred_proba: Optional[str], default=None
            Name(s) of the column(s) containing your model output. For binary classification, pass a single string
            refering to the model output column.
        """
        super().__init__(
            name='roc_auc',
            y_true=y_true,
            y_pred=y_pred,
            threshold=threshold,
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
        data = _remove_nans(data, (self.y_true, self.y_pred))

        y_true = data[self.y_true]
        y_pred = data[self.y_pred_proba]

        if y_true.nunique() <= 1:
            warnings.warn(
                f"'{self.y_true}' only contains a single class for chunk, cannot calculate {self.display_name}. "
                f"Returning NaN."
            )
            return np.NaN
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
        threshold: Threshold,
        y_pred_proba: Optional[str] = None,
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
        y_pred_proba: Optional[str], default=None
            Name(s) of the column(s) containing your model output. For binary classification, pass a single string
            refering to the model output column.
        """
        super().__init__(
            name='f1',
            y_true=y_true,
            y_pred=y_pred,
            threshold=threshold,
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
        data = _remove_nans(data, (self.y_true, self.y_pred))

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_true.nunique() <= 1:
            warnings.warn(
                f"'{self.y_true}' only contains a single class for chunk, cannot calculate {self.display_name}. "
                f"Returning NaN."
            )
            return np.NaN
        elif y_pred.nunique() <= 1:
            warnings.warn(
                f"'{self.y_pred}' only contains a single class for chunk, cannot calculate {self.display_name}. "
                f"Returning NaN."
            )
            return np.NaN
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
        threshold: Threshold,
        y_pred_proba: Optional[str] = None,
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
        y_pred_proba: Optional[str], default=None
            Name(s) of the column(s) containing your model output. For binary classification, pass a single string
            refering to the model output column.
        """
        super().__init__(
            name='precision',
            y_true=y_true,
            y_pred=y_pred,
            threshold=threshold,
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
        data = _remove_nans(data, (self.y_true, self.y_pred))

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_true.nunique() <= 1:
            warnings.warn(
                f"'{self.y_true}' only contains a single class for chunk, cannot calculate {self.display_name}. "
                f"Returning NaN."
            )
            return np.NaN
        elif y_pred.nunique() <= 1:
            warnings.warn(
                f"'{self.y_pred}' only contains a single class for chunk, cannot calculate {self.display_name}. "
                f"Returning NaN."
            )
            return np.NaN
        else:
            return precision_score(y_true, y_pred)

    def _sampling_error(self, data: pd.DataFrame):
        return precision_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='recall', use_case=ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationRecall(Metric):
    """Recall metric, also known as 'sensitivity'.

    Parameters
    ----------
    y_true: str
        The name of the column containing target values.
    y_pred: str
        The name of the column containing your model predictions.
    threshold: Threshold
        The Threshold instance that determines how the lower and upper threshold values will be calculated.
    y_pred_proba: Optional[str], default=None
        Name(s) of the column(s) containing your model output. For binary classification, pass a single string
        refering to the model output column.
    """

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        threshold: Threshold,
        y_pred_proba: Optional[str] = None,
        **kwargs,
    ):
        """Creates a new Recall instance."""
        super().__init__(
            name='recall',
            y_true=y_true,
            y_pred=y_pred,
            threshold=threshold,
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
        data = _remove_nans(data, (self.y_true, self.y_pred))

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_true.nunique() <= 1:
            warnings.warn(
                f"'{self.y_true}' only contains a single class for chunk, cannot calculate {self.display_name}. "
                f"Returning NaN."
            )
            return np.NaN
        elif y_pred.nunique() <= 1:
            warnings.warn(
                f"'{self.y_pred}' only contains a single class for chunk, cannot calculate {self.display_name}. "
                f"Returning NaN."
            )
            return np.NaN
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
        threshold: Threshold,
        y_pred_proba: Optional[str] = None,
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
        y_pred_proba: Optional[str], default=None
            Name(s) of the column(s) containing your model output. For binary classification, pass a single string
            refering to the model output column.
        """
        super().__init__(
            name='specificity',
            y_true=y_true,
            y_pred=y_pred,
            threshold=threshold,
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
        data = _remove_nans(data, (self.y_true, self.y_pred))

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_true.nunique() <= 1:
            warnings.warn(
                f"'{self.y_true}' only contains a single class for chunk, cannot calculate {self.display_name}. "
                f"Returning NaN."
            )
            return np.NaN
        elif y_pred.nunique() <= 1:
            warnings.warn(
                f"'{self.y_pred}' only contains a single class for chunk, cannot calculate {self.display_name}. "
                f"Returning NaN."
            )
            return np.NaN
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fp)

    def _sampling_error(self, data: pd.DataFrame):
        return specificity_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='accuracy', use_case=ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationAccuracy(Metric):
    """Accuracy metric.

    Parameters
    ----------
    y_true: str
        The name of the column containing target values.
    y_pred: str
        The name of the column containing your model predictions.
    threshold: Threshold
        The Threshold instance that determines how the lower and upper threshold values will be calculated.
    y_pred_proba: Optional[str], default=None
        Name(s) of the column(s) containing your model output. For binary classification, pass a single string
        refering to the model output column.
    """

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        threshold: Threshold,
        y_pred_proba: Optional[str] = None,
        **kwargs,
    ):
        """Creates a new Accuracy instance."""
        super().__init__(
            name='accuracy',
            y_true=y_true,
            y_pred=y_pred,
            threshold=threshold,
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
        data = _remove_nans(data, (self.y_true, self.y_pred))

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_true.nunique() <= 1:
            warnings.warn(
                f"'{self.y_true}' only contains a single class for chunk, cannot calculate {self.display_name}. "
                f"Returning NaN."
            )
            return np.NaN
        elif y_pred.nunique() <= 1:
            warnings.warn(
                f"'{self.y_pred}' only contains a single class for chunk, cannot calculate {self.display_name}. "
                f"Returning NaN."
            )
            return np.NaN
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return (tp + tn) / (tp + tn + fp + fn)

    def _sampling_error(self, data: pd.DataFrame):
        return accuracy_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='business_value', use_case=ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationBusinessValue(Metric):
    """Business Value metric."""

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        threshold: Threshold,
        business_value_matrix: Union[List, np.ndarray],
        normalize_business_value: Optional[str] = None,
        y_pred_proba: Optional[str] = None,
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
            A 2x2 matrix that specifies the value of each cell in the confusion matrix.
            The format of the business value matrix must be specified as [[value_of_TN, value_of_FP], \
            [value_of_FN, value_of_TP]]. Required when estimating the 'business_value' metric.
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
        data = _remove_nans(data, (self.y_true, self.y_pred))

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_true.shape[0] == 0:
            warnings.warn(f"'{self.y_true}' contains no data, cannot calculate business value. Returning NaN.")
            return np.NaN

        tp_value = self.business_value_matrix[1, 1]
        tn_value = self.business_value_matrix[0, 0]
        fp_value = self.business_value_matrix[0, 1]
        fn_value = self.business_value_matrix[1, 0]
        bv_array = np.array([[tn_value, fp_value], [fn_value, tp_value]])

        cm = confusion_matrix(y_true, y_pred)
        if self.normalize_business_value == 'per_prediction':
            with np.errstate(all="ignore"):
                cm = cm / cm.sum(axis=0, keepdims=True)
            cm = np.nan_to_num(cm)

        return (bv_array * cm).sum()

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return business_value_sampling_error(self._sampling_error_components, data)


@MetricFactory.register(metric='confusion_matrix', use_case=ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationConfusionMatrix(Metric):
    """Confusion Matrix metric."""

    def __init__(
        self,
        y_true: str,
        y_pred: str,
        threshold: Threshold,
        normalize_confusion_matrix: Optional[str] = None,
        y_pred_proba: Optional[str] = None,
        **kwargs,
    ):
        """Creates a new Confusion Matrix instance.

        Parameters
        ----------
        y_true: str
            The name of the column containing target values.
        y_pred: str
            The name of the column containing your model predictions.
        threshold: Threshold
            The Threshold instance that determines how the lower and upper threshold values will be calculated.
        normalize_confusion_matrix: Optional[str], default=None
            Determines how the confusion matrix will be normalized. Allowed values are None, 'all', 'true' and
            'predicted'.
        y_pred_proba: Optional[str], default=None
            Name(s) of the column(s) containing your model output. For binary classification, pass a single string
            refering to the model output column.
        """
        super().__init__(
            name='confusion_matrix',
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            threshold=threshold,
            components=[
                ('True Positive', 'true_positive'),
                ('True Negative', 'true_negative'),
                ('False Positive', 'false_positive'),
                ('False Negative', 'false_negative'),
            ],
            lower_threshold_limit=0,
        )

        self.upper_threshold_value_limit: Optional[float] = 1.0 if normalize_confusion_matrix else None
        self.normalize_confusion_matrix: Optional[str] = normalize_confusion_matrix
        # sampling error
        self._sampling_error_components: Tuple = ()

    def __str__(self):
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
        self._fit(reference_data)

        # Calculate alert thresholds
        reference_chunks = chunker.split(
            reference_data,
        )

        (
            self.true_positive_lower_threshold,
            self.true_positive_upper_threshold,
        ) = self._calculate_confusion_matrix_alert_thresholds(
            metric_name='true_positive',
            reference_chunks=reference_chunks,
        )
        (
            self.true_negative_lower_threshold,
            self.true_negative_upper_threshold,
        ) = self._calculate_confusion_matrix_alert_thresholds(
            metric_name='true_negative',
            reference_chunks=reference_chunks,
        )
        (
            self.false_positive_lower_threshold,
            self.false_positive_upper_threshold,
        ) = self._calculate_confusion_matrix_alert_thresholds(
            metric_name='false_positive',
            reference_chunks=reference_chunks,
        )
        (
            self.false_negative_lower_threshold,
            self.false_negative_upper_threshold,
        ) = self._calculate_confusion_matrix_alert_thresholds(
            metric_name='false_negative',
            reference_chunks=reference_chunks,
        )

        return

    def _calculate_confusion_matrix_alert_thresholds(
        self,
        metric_name: str,
        reference_chunks: List[Chunk],
    ) -> Tuple[Optional[float], Optional[float]]:
        if metric_name == 'true_positive':
            chunked_reference_metric = [self._calculate_true_positives(chunk.data) for chunk in reference_chunks]
        elif metric_name == 'true_negative':
            chunked_reference_metric = [self._calculate_true_negatives(chunk.data) for chunk in reference_chunks]
        elif metric_name == 'false_positive':
            chunked_reference_metric = [self._calculate_false_positives(chunk.data) for chunk in reference_chunks]
        elif metric_name == 'false_negative':
            chunked_reference_metric = [self._calculate_false_negatives(chunk.data) for chunk in reference_chunks]
        else:
            raise InvalidArgumentsException(f"could not calculate metric {metric_name}. invalid metric name")

        lower_threshold_value, upper_threshold_value = calculate_threshold_values(
            threshold=self.threshold,
            data=np.asarray(chunked_reference_metric),
            lower_threshold_value_limit=self.lower_threshold_value_limit,
            upper_threshold_value_limit=self.upper_threshold_value_limit,
            logger=self._logger,
            metric_name=self.display_name,
        )

        return lower_threshold_value, upper_threshold_value

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))

        self._true_positive_sampling_error_components = true_positive_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
            normalize_confusion_matrix=self.normalize_confusion_matrix,
        )
        self._true_negative_sampling_error_components = true_negative_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
            normalize_confusion_matrix=self.normalize_confusion_matrix,
        )
        self._false_positive_sampling_error_components = false_positive_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
            normalize_confusion_matrix=self.normalize_confusion_matrix,
        )
        self._false_negative_sampling_error_components = false_negative_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
            normalize_confusion_matrix=self.normalize_confusion_matrix,
        )

    def _calculate_true_positives(self, data: pd.DataFrame) -> float:
        _list_missing([self.y_true, self.y_pred], list(data.columns))
        data = _remove_nans(data, (self.y_true, self.y_pred))

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_true.empty or y_pred.empty:
            warnings.warn("Calculated true_positives contain NaN values.")
            return np.nan

        num_tp = np.sum(np.logical_and(y_pred, y_true))
        num_fn = np.sum(np.logical_and(np.logical_not(y_pred), y_true))
        num_fp = np.sum(np.logical_and(y_pred, np.logical_not(y_true)))

        if self.normalize_confusion_matrix is None:
            return num_tp
        elif self.normalize_confusion_matrix == 'true':
            return num_tp / (num_tp + num_fn)
        elif self.normalize_confusion_matrix == 'pred':
            return num_tp / (num_tp + num_fp)
        else:  # normalize_confusion_matrix == 'all'
            return num_tp / len(y_true)

    def _calculate_true_negatives(self, data: pd.DataFrame) -> float:
        _list_missing([self.y_true, self.y_pred], list(data.columns))
        data = _remove_nans(data, (self.y_true, self.y_pred))

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_true.empty or y_pred.empty:
            warnings.warn("Calculated true_negatives contain NaN values.")
            return np.nan

        num_tn = np.sum(np.logical_and(np.logical_not(y_pred), np.logical_not(y_true)))
        num_fn = np.sum(np.logical_and(np.logical_not(y_pred), y_true))
        num_fp = np.sum(np.logical_and(y_pred, np.logical_not(y_true)))

        if self.normalize_confusion_matrix is None:
            return num_tn
        elif self.normalize_confusion_matrix == 'true':
            return num_tn / (num_tn + num_fp)
        elif self.normalize_confusion_matrix == 'pred':
            return num_tn / (num_tn + num_fn)
        else:  # normalize_confusion_matrix == 'all'
            return num_tn / len(y_true)

    def _calculate_false_positives(self, data: pd.DataFrame) -> float:
        _list_missing([self.y_true, self.y_pred], list(data.columns))
        data = _remove_nans(data, (self.y_true, self.y_pred))

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_true.empty or y_pred.empty:
            warnings.warn("Calculated false_positives contain NaN values.")
            return np.nan

        num_fp = np.sum(np.logical_and(y_pred, np.logical_not(y_true)))
        num_tn = np.sum(np.logical_and(np.logical_not(y_pred), np.logical_not(y_true)))
        num_tp = np.sum(np.logical_and(y_pred, y_true))

        if self.normalize_confusion_matrix is None:
            return num_fp
        elif self.normalize_confusion_matrix == 'true':
            return num_fp / (num_fp + num_tn)
        elif self.normalize_confusion_matrix == 'pred':
            return num_fp / (num_fp + num_tp)
        else:  # normalize_confusion_matrix == 'all'
            return num_fp / len(y_true)

    def _calculate_false_negatives(self, data: pd.DataFrame) -> float:
        _list_missing([self.y_true, self.y_pred], list(data.columns))
        data = _remove_nans(data, (self.y_true, self.y_pred))

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_true.empty or y_pred.empty:
            warnings.warn(f"'{self.y_true}' contains no data, cannot calculate {self.display_name}. Returning NaN.")
            return np.NaN

        num_fn = np.sum(np.logical_and(np.logical_not(y_pred), y_true))
        num_tn = np.sum(np.logical_and(np.logical_not(y_pred), np.logical_not(y_true)))
        num_tp = np.sum(np.logical_and(y_pred, y_true))

        if self.normalize_confusion_matrix is None:
            return num_fn
        elif self.normalize_confusion_matrix == 'true':
            return num_fn / (num_fn + num_tp)
        elif self.normalize_confusion_matrix == 'pred':
            return num_fn / (num_fn + num_tn)
        else:  # normalize_confusion_matrix == 'all'
            return num_fn / len(y_true)

    def get_true_pos_info(self, chunk_data: pd.DataFrame) -> Dict:
        """Returns a dictionary containing infomation about the true positives for a given chunk.

        Parameters
        ----------
        chunk_data : pd.DataFrame
            A pandas dataframe containing the data for a given chunk.

        Returns
        -------
        true_pos_info : Dict
            A dictionary of true positive's information and its value pairs.
        """

        column_name = 'true_positive'

        true_pos_info: Dict[str, Any] = {}

        realized_tp = self._calculate_true_positives(chunk_data)  # in this function, check if there are
        sampling_error_tp = true_positive_sampling_error(self._true_positive_sampling_error_components, chunk_data)

        true_pos_info[f'{column_name}_sampling_error'] = sampling_error_tp
        true_pos_info[f'{column_name}'] = realized_tp
        true_pos_info[f'{column_name}_upper_threshold'] = self.true_positive_upper_threshold
        true_pos_info[f'{column_name}_lower_threshold'] = self.true_positive_lower_threshold
        true_pos_info[f'{column_name}_alert'] = (
            self.true_positive_lower_threshold is not None and self.true_positive_lower_threshold > realized_tp
        ) or (self.true_positive_upper_threshold is not None and self.true_positive_upper_threshold < realized_tp)

        return true_pos_info

    def get_true_neg_info(self, chunk_data: pd.DataFrame) -> Dict:
        """Returns a dictionary containing infomation about the true negatives for a given chunk.

        Parameters
        ----------
        chunk_data : pd.DataFrame
            A pandas dataframe containing the data for a given chunk.

        Returns
        -------
        true_neg_info : Dict
            A dictionary of true negative's information and its value pairs.
        """

        column_name = 'true_negative'

        true_neg_info: Dict[str, Any] = {}

        realized_tn = self._calculate_true_negatives(chunk_data)
        sampling_error_tn = true_negative_sampling_error(self._true_negative_sampling_error_components, chunk_data)

        true_neg_info[f'{column_name}_sampling_error'] = sampling_error_tn
        true_neg_info[f'{column_name}'] = realized_tn
        true_neg_info[f'{column_name}_upper_threshold'] = self.true_negative_upper_threshold
        true_neg_info[f'{column_name}_lower_threshold'] = self.true_negative_lower_threshold
        true_neg_info[f'{column_name}_alert'] = (
            (self.true_negative_lower_threshold is not None and self.true_negative_lower_threshold > realized_tn)
        ) or (self.true_negative_upper_threshold is not None and self.true_negative_upper_threshold < realized_tn)

        return true_neg_info

    def get_false_pos_info(self, chunk_data: pd.DataFrame) -> Dict:
        """Returns a dictionary containing infomation about the false positives for a given chunk.

        Parameters
        ----------
        chunk_data : pd.DataFrame
            A pandas dataframe containing the data for a given chunk.

        Returns
        -------
        false_pos_info : Dict
            A dictionary of false positive's information and its value pairs.
        """

        column_name = 'false_positive'

        false_pos_info: Dict[str, Any] = {}

        realized_fp = self._calculate_false_positives(chunk_data)
        sampling_error_fp = false_positive_sampling_error(self._false_positive_sampling_error_components, chunk_data)

        false_pos_info[f'{column_name}_sampling_error'] = sampling_error_fp
        false_pos_info[f'{column_name}'] = realized_fp
        false_pos_info[f'{column_name}_upper_threshold'] = self.false_positive_upper_threshold
        false_pos_info[f'{column_name}_lower_threshold'] = self.false_positive_lower_threshold
        false_pos_info[f'{column_name}_alert'] = (
            self.false_positive_lower_threshold is not None and self.false_positive_lower_threshold > realized_fp
        ) or (self.false_positive_upper_threshold is not None and self.false_positive_upper_threshold < realized_fp)

        return false_pos_info

    def get_false_neg_info(self, chunk_data: pd.DataFrame) -> Dict:
        """Returns a dictionary containing infomation about the false negatives for a given chunk.

        Parameters
        ----------
        chunk_data : pd.DataFrame
            A pandas dataframe containing the data for a given chunk.

        Returns
        -------
        false_neg_info : Dict
            A dictionary of false negative's information and its value pairs.
        """

        column_name = 'false_negative'

        false_neg_info: Dict[str, Any] = {}

        realized_fn = self._calculate_false_negatives(chunk_data)
        sampling_error_fn = false_negative_sampling_error(self._false_negative_sampling_error_components, chunk_data)

        false_neg_info[f'{column_name}_sampling_error'] = sampling_error_fn
        false_neg_info[f'{column_name}'] = realized_fn
        false_neg_info[f'{column_name}_upper_threshold'] = self.false_negative_upper_threshold
        false_neg_info[f'{column_name}_lower_threshold'] = self.false_negative_lower_threshold
        false_neg_info[f'{column_name}_alert'] = (
            self.false_negative_lower_threshold is not None and self.false_negative_lower_threshold > realized_fn
        ) or (self.false_negative_upper_threshold is not None and self.false_negative_upper_threshold < realized_fn)

        return false_neg_info

    def get_chunk_record(self, chunk_data: pd.DataFrame) -> Dict:
        """Returns a dictionary containing the conduction matrix values for a given chunk.

        Parameters
        ----------
        chunk_data : pd.DataFrame
            A pandas dataframe containing the data for a given chunk.

        Returns
        -------
            chunk_record : Dict
                A dictionary of confusion matrix metrics, value pairs.
        """
        chunk_record = {}

        true_pos_info = self.get_true_pos_info(chunk_data)
        chunk_record.update(true_pos_info)

        true_neg_info = self.get_true_neg_info(chunk_data)
        chunk_record.update(true_neg_info)

        false_pos_info = self.get_false_pos_info(chunk_data)
        chunk_record.update(false_pos_info)

        false_neg_info = self.get_false_neg_info(chunk_data)
        chunk_record.update(false_neg_info)

        return chunk_record

    def _calculate(self, data: pd.DataFrame):
        pass

    def _sampling_error(self, data: pd.DataFrame):
        pass
