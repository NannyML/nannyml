"""Shared support for optional instance-level metric contributions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from nannyml.chunk import Chunk
from nannyml.exceptions import InvalidArgumentsException

INSTANCE_IDENTIFIER_COLUMN_NAME = '__nml_instance_identifier'
DEFAULT_IDENTIFIER_OUTPUT_COLUMN_NAME = 'identifier'
INSTANCE_METRIC_SUFFIX = '_contribution'
INSTANCE_METADATA_COLUMN_NAMES = (
    'chunk_key',
    'chunk_index',
    'chunk_start_index',
    'chunk_end_index',
    'chunk_start_date',
    'chunk_end_date',
    'period',
)

InstanceMetricSelection = Union[bool, Sequence[str]]


def resolve_instance_metric_selection(
    requested: InstanceMetricSelection,
    supported_metrics: Sequence[str],
) -> List[str]:
    """Resolve the public instance metric selector against calculator-supported names."""
    if requested is False:
        return []
    if requested is True:
        return list(supported_metrics)
    if isinstance(requested, str) or not isinstance(requested, Sequence):
        raise InvalidArgumentsException("include_instance_metrics must be False, True, or a list of metric names.")

    selected = list(requested)
    if len(selected) != len(set(selected)):
        raise InvalidArgumentsException('include_instance_metrics contains duplicate metric names.')

    unsupported = [metric for metric in selected if metric not in supported_metrics]
    if unsupported:
        raise InvalidArgumentsException(
            f"instance-level metrics {unsupported} are not supported. Supported metrics are {list(supported_metrics)}."
        )
    return selected


def prepare_data_for_instance_metrics(
    data: pd.DataFrame,
    identifier_column_name: Optional[str],
) -> Tuple[pd.DataFrame, str]:
    """Copy data and attach a stable internal row identifier used while chunking."""
    if INSTANCE_IDENTIFIER_COLUMN_NAME in data.columns:
        raise InvalidArgumentsException(
            f"data contains reserved column name '{INSTANCE_IDENTIFIER_COLUMN_NAME}'. Please rename this column."
        )

    prepared = data.copy(deep=True)
    if identifier_column_name is None:
        identifiers = prepared.index
        output_column_name = DEFAULT_IDENTIFIER_OUTPUT_COLUMN_NAME
    else:
        if identifier_column_name not in prepared.columns:
            raise InvalidArgumentsException(
                f"identifier column '{identifier_column_name}' not found in data columns: {list(prepared.columns)}."
            )
        identifiers = prepared[identifier_column_name]
        output_column_name = identifier_column_name

    if output_column_name in INSTANCE_METADATA_COLUMN_NAMES or output_column_name.endswith(INSTANCE_METRIC_SUFFIX):
        raise InvalidArgumentsException(
            f"identifier column name '{output_column_name}' cannot be used in the instance-level output."
        )

    if isinstance(identifiers, pd.MultiIndex):
        if identifiers.to_frame(index=False).isna().to_numpy().any():
            raise InvalidArgumentsException('instance identifiers must not contain missing values.')
        identifier_series = pd.Series(list(identifiers), index=prepared.index, dtype=object)
    else:
        identifier_series = pd.Series(identifiers, index=prepared.index)
        if identifier_series.isna().any():
            raise InvalidArgumentsException('instance identifiers must not contain missing values.')
    if not identifier_series.is_unique:
        raise InvalidArgumentsException('instance identifiers must be unique within the provided data.')

    prepared[INSTANCE_IDENTIFIER_COLUMN_NAME] = identifier_series.to_numpy()
    return prepared, output_column_name


def build_instance_chunk_record(
    chunk: Chunk,
    identifier_column_name: str,
    period: str,
    contributions: Dict[str, Union[pd.Series, np.ndarray, Sequence[float]]],
) -> pd.DataFrame:
    """Build the flat, merge-friendly instance table for a single chunk."""
    record = pd.DataFrame(
        {
            identifier_column_name: chunk.data[INSTANCE_IDENTIFIER_COLUMN_NAME].to_numpy(),
            'chunk_key': chunk.key,
            'chunk_index': chunk.chunk_index,
            'chunk_start_index': chunk.start_index,
            'chunk_end_index': chunk.end_index,
            'chunk_start_date': chunk.start_datetime,
            'chunk_end_date': chunk.end_datetime,
            'period': period,
        }
    )
    for column_name, values in contributions.items():
        if not column_name.endswith(INSTANCE_METRIC_SUFFIX):
            raise InvalidArgumentsException(
                f"instance metric column '{column_name}' must end with '{INSTANCE_METRIC_SUFFIX}'."
            )
        values_array = np.asarray(values)
        if len(values_array) != len(record):
            raise InvalidArgumentsException(
                f"instance metric column '{column_name}' contains {len(values_array)} values for {len(record)} rows."
            )
        record[column_name] = values_array
    return record


def combine_instance_chunk_records(records: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Combine chunk instance records while retaining a stable empty-table behavior."""
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def _with_missing_rows(values: np.ndarray, valid: pd.Series, length: int) -> np.ndarray:
    result = np.full(length, np.nan, dtype=float)
    result[np.flatnonzero(valid.to_numpy())] = values
    return result


def _binary_auc_contributions(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    positives = scores[y_true == 1]
    negatives = scores[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return np.full(len(y_true), np.nan)

    sorted_positives = np.sort(positives)
    sorted_negatives = np.sort(negatives)
    result = np.empty(len(y_true), dtype=float)
    for index, (target, score) in enumerate(zip(y_true, scores)):
        if target == 1:
            lower = np.searchsorted(sorted_negatives, score, side='left')
            equal = np.searchsorted(sorted_negatives, score, side='right') - lower
            result[index] = (lower + 0.5 * equal) / len(sorted_negatives)
        else:
            higher = len(sorted_positives) - np.searchsorted(sorted_positives, score, side='right')
            equal = np.searchsorted(sorted_positives, score, side='right') - np.searchsorted(
                sorted_positives, score, side='left'
            )
            result[index] = (higher + 0.5 * equal) / len(sorted_positives)
    return result


def mann_whitney_positive_contributions(reference_scores: np.ndarray, analysis_scores: np.ndarray) -> np.ndarray:
    """Return each analysis row's concordance against all reference scores."""
    if len(reference_scores) == 0:
        return np.full(len(analysis_scores), np.nan)
    sorted_reference = np.sort(reference_scores)
    lower = np.searchsorted(sorted_reference, analysis_scores, side='left')
    equal = np.searchsorted(sorted_reference, analysis_scores, side='right') - lower
    return (lower + 0.5 * equal) / len(sorted_reference)


def _binary_ap_contributions(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Allocate AP credit and ranking penalties symmetrically across pair endpoints."""
    positive_count = int(np.sum(y_true == 1))
    if positive_count == 0:
        return np.full(len(y_true), np.nan)

    unique_scores = np.unique(scores)[::-1]
    group_indices = [np.flatnonzero(scores == score) for score in unique_scores]
    group_positives = np.asarray([np.sum(y_true[indices] == 1) for indices in group_indices], dtype=float)
    group_negatives = np.asarray([np.sum(y_true[indices] == 0) for indices in group_indices], dtype=float)
    cumulative_total = np.cumsum(group_positives + group_negatives)
    cumulative_negatives = np.cumsum(group_negatives)

    positive_penalty = cumulative_negatives / (positive_count * cumulative_total)
    negative_penalty = np.cumsum((group_positives / (positive_count * cumulative_total))[::-1])[::-1]

    result = np.zeros(len(y_true), dtype=float)
    for group_index, indices in enumerate(group_indices):
        positive_indices = indices[y_true[indices] == 1]
        negative_indices = indices[y_true[indices] == 0]
        result[positive_indices] = (1.0 / positive_count) - 0.5 * positive_penalty[group_index]
        result[negative_indices] = -0.5 * negative_penalty[group_index]
    return result


def _classification_cell_contributions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[Any],
    normalize: Optional[str],
) -> Dict[str, np.ndarray]:
    contributions: Dict[str, np.ndarray] = {}
    row_count = len(y_true)
    for true_class in classes:
        for predicted_class in classes:
            indicator = ((y_true == true_class) & (y_pred == predicted_class)).astype(float)
            if normalize is None:
                values = indicator
            elif normalize == 'all':
                values = indicator / row_count
            elif normalize == 'true':
                denominator = np.sum(y_true == true_class)
                values = indicator / denominator if denominator else np.full(row_count, np.nan)
            else:  # ``pred`` in NannyML and ``predicted`` in the underlying sklearn API.
                denominator = np.sum(y_pred == predicted_class)
                values = indicator / denominator if denominator else np.full(row_count, np.nan)
            contributions[f'true_{true_class}_pred_{predicted_class}{INSTANCE_METRIC_SUFFIX}'] = values
    return contributions


def _binary_classification_contributions(metric: Any, data: pd.DataFrame) -> Dict[str, np.ndarray]:
    metric_name = metric.name
    required = (
        [metric.y_true, metric.y_pred_proba]
        if metric_name in ['roc_auc', 'average_precision']
        else [
            metric.y_true,
            metric.y_pred,
        ]
    )
    valid = data[required].notna().all(axis=1)
    clean = data.loc[valid, required]
    if clean.empty:
        columns = metric.column_names if metric_name == 'confusion_matrix' else [metric.column_name]
        return {f'{column}{INSTANCE_METRIC_SUFFIX}': np.full(len(data), np.nan) for column in columns}

    y_true = clean[metric.y_true].to_numpy()
    if metric_name == 'roc_auc':
        values = _binary_auc_contributions(y_true, clean[metric.y_pred_proba].to_numpy(dtype=float))
    elif metric_name == 'average_precision':
        values = _binary_ap_contributions(y_true, clean[metric.y_pred_proba].to_numpy(dtype=float))
    else:
        y_pred = clean[metric.y_pred].to_numpy()
        if metric_name == 'accuracy':
            values = (y_true == y_pred).astype(float)
        elif metric_name == 'precision':
            values = np.where(y_pred == 1, (y_true == 1).astype(float), np.nan)
        elif metric_name == 'recall':
            values = np.where(y_true == 1, (y_pred == 1).astype(float), np.nan)
        elif metric_name == 'specificity':
            values = np.where(y_true == 0, (y_pred == 0).astype(float), np.nan)
        elif metric_name == 'f1':
            applicable = (y_true == 1) | (y_pred == 1)
            values = np.where(applicable, ((y_true == 1) & (y_pred == 1)).astype(float), np.nan)
        elif metric_name == 'business_value':
            values = metric.business_value_matrix[y_true.astype(int), y_pred.astype(int)].astype(float)
            if metric.normalize_business_value == 'per_prediction':
                predicted_counts = {predicted: np.sum(y_pred == predicted) for predicted in np.unique(y_pred)}
                values = np.asarray(
                    [value / predicted_counts[predicted] for value, predicted in zip(values, y_pred)], dtype=float
                )
        elif metric_name == 'confusion_matrix':
            clean_contributions = _classification_cell_contributions(
                y_true, y_pred, [1, 0], metric.normalize_confusion_matrix
            )
            # Binary public component names use semantic names rather than class-pair names.
            renamed = {
                f'true_positive{INSTANCE_METRIC_SUFFIX}': clean_contributions[f'true_1_pred_1{INSTANCE_METRIC_SUFFIX}'],
                f'true_negative{INSTANCE_METRIC_SUFFIX}': clean_contributions[f'true_0_pred_0{INSTANCE_METRIC_SUFFIX}'],
                f'false_positive{INSTANCE_METRIC_SUFFIX}': clean_contributions[
                    f'true_0_pred_1{INSTANCE_METRIC_SUFFIX}'
                ],
                f'false_negative{INSTANCE_METRIC_SUFFIX}': clean_contributions[
                    f'true_1_pred_0{INSTANCE_METRIC_SUFFIX}'
                ],
            }
            return {
                column: _with_missing_rows(component_values, valid, len(data))
                for column, component_values in renamed.items()
            }
        else:  # pragma: no cover - factory validation keeps this unreachable
            raise InvalidArgumentsException(f"no instance contribution implementation for metric '{metric_name}'.")

    return {f'{metric.column_name}{INSTANCE_METRIC_SUFFIX}': _with_missing_rows(values, valid, len(data))}


def _regression_contributions(metric: Any, data: pd.DataFrame) -> Dict[str, np.ndarray]:
    valid = data[[metric.y_true, metric.y_pred]].notna().all(axis=1)
    y_true = data.loc[valid, metric.y_true].to_numpy(dtype=float)
    y_pred = data.loc[valid, metric.y_pred].to_numpy(dtype=float)
    if metric.name == 'mae':
        values = np.abs(y_true - y_pred)
    elif metric.name == 'mape':
        values = np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), np.finfo(np.float64).eps)
    elif metric.name in ['mse', 'rmse']:
        values = np.square(y_true - y_pred)
    elif metric.name in ['msle', 'rmsle']:
        values = np.square(np.log1p(y_true) - np.log1p(y_pred))
    else:  # pragma: no cover - factory validation keeps this unreachable
        raise InvalidArgumentsException(f"no instance contribution implementation for metric '{metric.name}'.")
    return {f'{metric.column_name}{INSTANCE_METRIC_SUFFIX}': _with_missing_rows(values, valid, len(data))}


def _multiclass_classification_contributions(metric: Any, data: pd.DataFrame) -> Dict[str, np.ndarray]:
    metric_name = metric.name
    probability_columns = list(metric.y_pred_proba.values()) if isinstance(metric.y_pred_proba, dict) else []
    required = (
        [metric.y_true] + probability_columns
        if metric_name in ['roc_auc', 'average_precision']
        else [
            metric.y_true,
            metric.y_pred,
        ]
    )
    valid = data[required].notna().all(axis=1)
    clean = data.loc[valid, required]
    metric_classes = getattr(metric, 'classes', None)
    classes = list(metric_classes) if metric_classes is not None else sorted(metric.y_pred_proba.keys())
    y_true = clean[metric.y_true].to_numpy()

    if metric_name == 'confusion_matrix':
        y_pred = clean[metric.y_pred].to_numpy()
        clean_contributions = _classification_cell_contributions(
            y_true, y_pred, classes, metric.normalize_confusion_matrix
        )
        return {
            column: _with_missing_rows(component_values, valid, len(data))
            for column, component_values in clean_contributions.items()
        }

    if metric_name in ['roc_auc', 'average_precision']:
        class_contributions = []
        for clazz in classes:
            binary_target = (y_true == clazz).astype(int)
            score = clean[metric.y_pred_proba[clazz]].to_numpy(dtype=float)
            class_contributions.append(
                _binary_auc_contributions(binary_target, score)
                if metric_name == 'roc_auc'
                else _binary_ap_contributions(binary_target, score)
            )
        if metric_name == 'roc_auc':
            values = np.nanmean(np.vstack(class_contributions), axis=0)
        else:
            values = np.nansum(np.vstack(class_contributions), axis=0) / len(classes)
    else:
        y_pred = clean[metric.y_pred].to_numpy()
        if metric_name == 'accuracy':
            values = (y_true == y_pred).astype(float)
        elif metric_name == 'business_value':
            class_indices = {clazz: index for index, clazz in enumerate(classes)}
            values = np.asarray(
                [
                    metric.business_value_matrix[class_indices[actual], class_indices[predicted]]
                    for actual, predicted in zip(y_true, y_pred)
                ],
                dtype=float,
            )
            if metric.normalize_business_value == 'per_prediction':
                predicted_counts = {predicted: np.sum(y_pred == predicted) for predicted in np.unique(y_pred)}
                values = np.asarray(
                    [value / predicted_counts[predicted] for value, predicted in zip(values, y_pred)], dtype=float
                )
        else:
            values = np.zeros(len(clean), dtype=float)
            for clazz in classes:
                true_positive = (y_true == clazz) & (y_pred == clazz)
                if metric_name == 'precision':
                    denominator = np.sum(y_pred == clazz)
                    if denominator:
                        values += true_positive.astype(float) / (len(classes) * denominator)
                elif metric_name == 'recall':
                    denominator = np.sum(y_true == clazz)
                    if denominator:
                        values += true_positive.astype(float) / (len(classes) * denominator)
                elif metric_name == 'specificity':
                    negative = y_true != clazz
                    denominator = np.sum(negative)
                    if denominator:
                        values += (negative & (y_pred != clazz)).astype(float) / (len(classes) * denominator)
                elif metric_name == 'f1':
                    false_positive = (y_true != clazz) & (y_pred == clazz)
                    false_negative = (y_true == clazz) & (y_pred != clazz)
                    denominator = 2 * np.sum(true_positive) + np.sum(false_positive) + np.sum(false_negative)
                    if denominator:
                        values += 2 * true_positive.astype(float) / (len(classes) * denominator)
                else:  # pragma: no cover
                    raise InvalidArgumentsException(
                        f"no instance contribution implementation for metric '{metric_name}'."
                    )

    return {f'{metric.column_name}{INSTANCE_METRIC_SUFFIX}': _with_missing_rows(values, valid, len(data))}


def realized_metric_contributions(metric: Any, data: pd.DataFrame, problem_type: Any) -> Dict[str, np.ndarray]:
    """Calculate per-row contributions for a realized performance metric."""
    problem_type_value = getattr(problem_type, 'value', problem_type)
    if problem_type_value == 'regression':
        return _regression_contributions(metric, data)
    if problem_type_value == 'classification_binary':
        return _binary_classification_contributions(metric, data)
    if problem_type_value == 'classification_multiclass':
        return _multiclass_classification_contributions(metric, data)
    raise InvalidArgumentsException(f"unsupported problem type '{problem_type_value}'.")


def _soft_auc_contributions(probabilities: np.ndarray, scores: np.ndarray) -> np.ndarray:
    positive_weight = float(np.sum(probabilities))
    negative_probabilities = 1.0 - probabilities
    negative_weight = float(np.sum(negative_probabilities))
    if positive_weight == 0 or negative_weight == 0:
        return np.full(len(probabilities), np.nan)

    unique_scores = np.unique(scores)
    positive_by_score = np.asarray([np.sum(probabilities[scores == score]) for score in unique_scores])
    negative_by_score = np.asarray([np.sum(negative_probabilities[scores == score]) for score in unique_scores])
    negative_below = np.r_[0.0, np.cumsum(negative_by_score)[:-1]]
    positive_above = np.sum(positive_by_score) - np.cumsum(positive_by_score)
    score_index = {score: index for index, score in enumerate(unique_scores)}

    result = np.empty(len(probabilities), dtype=float)
    for row_index, (probability, score) in enumerate(zip(probabilities, scores)):
        group_index = score_index[score]
        positive_concordance = (negative_below[group_index] + 0.5 * negative_by_score[group_index]) / negative_weight
        negative_concordance = (positive_above[group_index] + 0.5 * positive_by_score[group_index]) / positive_weight
        opportunity = probability * negative_weight + (1.0 - probability) * positive_weight
        result[row_index] = (
            probability * negative_weight * positive_concordance
            + (1.0 - probability) * positive_weight * negative_concordance
        ) / opportunity

    # CBPE's existing estimator rounds cumulative counts and gives tied rows a stable sort order.
    # A constant reconciliation offset preserves the Mann-Whitney row ordering while making the
    # contribution mean reproduce that exact estimator.
    from nannyml.performance_estimation.confidence_based.metrics import estimate_roc_auc

    exact_value = estimate_roc_auc(probabilities, scores)
    if np.isfinite(exact_value):
        result += exact_value - float(np.mean(result))
    return result


def _soft_ap_value(probabilities: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)[::-1]
    sorted_probabilities = probabilities[order]
    true_positives = np.round(np.cumsum(sorted_probabilities), 5)
    if len(true_positives) == 0 or true_positives[-1] == 0:
        return np.nan
    precision = true_positives / np.arange(1, len(true_positives) + 1)
    recall_steps = np.diff(np.r_[0.0, true_positives]) / true_positives[-1]
    return float(np.sum(recall_steps * precision))


def _soft_ap_contributions(probabilities: np.ndarray, scores: np.ndarray) -> np.ndarray:
    positive_weight = float(np.sum(probabilities))
    if positive_weight == 0:
        return np.full(len(probabilities), np.nan)

    order = np.argsort(scores)[::-1]
    sorted_probabilities = probabilities[order]
    sorted_negative_probabilities = 1.0 - sorted_probabilities
    ranks = np.arange(1, len(probabilities) + 1, dtype=float)
    cumulative_negative = np.cumsum(sorted_negative_probabilities)

    positive_penalty = sorted_probabilities * cumulative_negative / (positive_weight * ranks)
    negative_penalty = (
        sorted_negative_probabilities * np.cumsum((sorted_probabilities / (positive_weight * ranks))[::-1])[::-1]
    )
    sorted_contributions = sorted_probabilities / positive_weight - 0.5 * (positive_penalty + negative_penalty)

    # CBPE rounds cumulative expected counts. Preserve that existing behavior exactly.
    exact_value = _soft_ap_value(probabilities, scores)
    residual = exact_value - float(np.sum(sorted_contributions))
    sorted_contributions += residual * sorted_probabilities / positive_weight

    result = np.empty(len(probabilities), dtype=float)
    result[order] = sorted_contributions
    return result


def _binary_cbpe_contributions(metric: Any, data: pd.DataFrame) -> Dict[str, np.ndarray]:
    ranking_metric = metric.name in ['roc_auc', 'average_precision']
    required = (
        [metric.y_pred_proba, metric.uncalibrated_y_pred_proba]
        if ranking_metric
        else [metric.y_pred_proba, metric.y_pred]
    )
    valid = data[required].notna().all(axis=1)
    clean = data.loc[valid, required]
    probabilities = clean[metric.y_pred_proba].to_numpy(dtype=float)

    if metric.name == 'roc_auc':
        values = _soft_auc_contributions(probabilities, clean[metric.uncalibrated_y_pred_proba].to_numpy(dtype=float))
    elif metric.name == 'average_precision':
        values = _soft_ap_contributions(probabilities, clean[metric.uncalibrated_y_pred_proba].to_numpy(dtype=float))
    else:
        y_pred = clean[metric.y_pred].to_numpy()
        if metric.name == 'accuracy':
            values = np.where(y_pred == 1, probabilities, 1.0 - probabilities)
        elif metric.name == 'precision':
            values = (
                np.where(y_pred == 1, probabilities, np.nan)
                if np.any(y_pred == 1)
                else np.zeros(len(clean), dtype=float)
            )
        elif metric.name == 'recall':
            values = np.where(y_pred == 1, probabilities, -probabilities)
        elif metric.name == 'specificity':
            values = np.where(y_pred == 0, 1.0 - probabilities, -(1.0 - probabilities))
        elif metric.name == 'f1':
            values = np.where(y_pred == 1, probabilities, -probabilities)
        elif metric.name == 'confusion_matrix':
            cell_values = {
                'true_positive': np.where(y_pred == 1, probabilities, 0.0),
                'true_negative': np.where(y_pred == 0, 1.0 - probabilities, 0.0),
                'false_positive': np.where(y_pred == 1, 1.0 - probabilities, 0.0),
                'false_negative': np.where(y_pred == 0, probabilities, 0.0),
            }
            positive_weight = np.sum(probabilities)
            negative_weight = np.sum(1.0 - probabilities)
            predicted_positive = np.sum(y_pred == 1)
            predicted_negative = np.sum(y_pred == 0)
            for name, component_values in cell_values.items():
                if metric.normalize_confusion_matrix == 'all':
                    cell_values[name] = component_values / len(clean)
                elif metric.normalize_confusion_matrix == 'true':
                    denominator = positive_weight if name in ['true_positive', 'false_negative'] else negative_weight
                    cell_values[name] = component_values / denominator
                elif metric.normalize_confusion_matrix == 'pred':
                    denominator = (
                        predicted_positive if name in ['true_positive', 'false_positive'] else predicted_negative
                    )
                    cell_values[name] = component_values / denominator
            return {
                f'{name}{INSTANCE_METRIC_SUFFIX}': _with_missing_rows(component_values, valid, len(data))
                for name, component_values in cell_values.items()
            }
        elif metric.name == 'business_value':
            values = np.asarray(
                [
                    probability * metric.business_value_matrix[1, int(predicted)]
                    + (1.0 - probability) * metric.business_value_matrix[0, int(predicted)]
                    for probability, predicted in zip(probabilities, y_pred)
                ]
            )
            if metric.normalize_business_value == 'per_prediction':
                predicted_counts = {predicted: np.sum(y_pred == predicted) for predicted in np.unique(y_pred)}
                values = np.asarray([value / predicted_counts[predicted] for value, predicted in zip(values, y_pred)])
        else:  # pragma: no cover
            raise InvalidArgumentsException(f"no instance contribution implementation for metric '{metric.name}'.")

    return {f'{metric.column_name}{INSTANCE_METRIC_SUFFIX}': _with_missing_rows(values, valid, len(data))}


def _multiclass_cbpe_contributions(metric: Any, data: pd.DataFrame) -> Dict[str, np.ndarray]:  # noqa: C901
    classes = sorted(metric.y_pred_proba.keys())
    probability_columns = [metric.y_pred_proba[clazz] for clazz in classes]
    ranking_metric = metric.name in ['roc_auc', 'average_precision']
    uncalibrated_columns = [f'uncalibrated_{column}' for column in probability_columns]
    required = probability_columns + (uncalibrated_columns if ranking_metric else [metric.y_pred])
    valid = data[required].notna().all(axis=1)
    clean = data.loc[valid, required]
    probabilities = clean[probability_columns].to_numpy(dtype=float)

    if metric.name in ['roc_auc', 'average_precision']:
        class_contributions = []
        for class_index, _ in enumerate(classes):
            scores = clean[uncalibrated_columns[class_index]].to_numpy(dtype=float)
            class_contributions.append(
                _soft_auc_contributions(probabilities[:, class_index], scores)
                if metric.name == 'roc_auc'
                else _soft_ap_contributions(probabilities[:, class_index], scores)
            )
        if metric.name == 'roc_auc':
            values = np.nanmean(np.vstack(class_contributions), axis=0)
        else:
            values = np.nansum(np.vstack(class_contributions), axis=0) / len(classes)
    else:
        y_pred = clean[metric.y_pred].to_numpy()
        predicted_indices = np.asarray([classes.index(predicted) for predicted in y_pred])
        if metric.name == 'accuracy':
            values = probabilities[np.arange(len(clean)), predicted_indices]
        elif metric.name == 'business_value':
            values = np.asarray(
                [
                    np.dot(probability_row, metric.business_value_matrix[:, predicted_index])
                    for probability_row, predicted_index in zip(probabilities, predicted_indices)
                ]
            )
            if metric.normalize_business_value == 'per_prediction':
                predicted_counts = {predicted: np.sum(y_pred == predicted) for predicted in np.unique(y_pred)}
                values = np.asarray([value / predicted_counts[predicted] for value, predicted in zip(values, y_pred)])
        elif metric.name == 'confusion_matrix':
            contributions = {}
            for true_index, true_class in enumerate(classes):
                for predicted_index, predicted_class in enumerate(classes):
                    component_values = np.where(predicted_indices == predicted_index, probabilities[:, true_index], 0.0)
                    if metric.normalize_confusion_matrix == 'all':
                        component_values = component_values / len(clean)
                    elif metric.normalize_confusion_matrix == 'true':
                        component_values = component_values / np.sum(probabilities[:, true_index])
                    elif metric.normalize_confusion_matrix == 'pred':
                        component_values = component_values / np.sum(predicted_indices == predicted_index)
                    contributions[
                        f'true_{true_class}_pred_{predicted_class}{INSTANCE_METRIC_SUFFIX}'
                    ] = _with_missing_rows(component_values, valid, len(data))
            return contributions
        else:
            values = np.zeros(len(clean), dtype=float)
            for class_index, _ in enumerate(classes):
                predicted_positive = predicted_indices == class_index
                probability = probabilities[:, class_index]
                true_positive = np.where(predicted_positive, probability, 0.0)
                false_positive = np.where(predicted_positive, 1.0 - probability, 0.0)
                false_negative = np.where(~predicted_positive, probability, 0.0)
                true_negative = np.where(~predicted_positive, 1.0 - probability, 0.0)
                if metric.name == 'precision':
                    denominator = np.sum(predicted_positive)
                    if denominator:
                        values += true_positive / (len(classes) * denominator)
                elif metric.name == 'recall':
                    denominator = np.sum(true_positive + false_negative)
                    if denominator:
                        values += true_positive / (len(classes) * denominator)
                elif metric.name == 'specificity':
                    denominator = np.sum(true_negative + false_positive)
                    if denominator:
                        values += true_negative / (len(classes) * denominator)
                elif metric.name == 'f1':
                    denominator = np.sum(true_positive + 0.5 * (false_positive + false_negative))
                    if denominator:
                        values += true_positive / (len(classes) * denominator)
                else:  # pragma: no cover
                    raise InvalidArgumentsException(
                        f"no instance contribution implementation for metric '{metric.name}'."
                    )

    return {f'{metric.column_name}{INSTANCE_METRIC_SUFFIX}': _with_missing_rows(values, valid, len(data))}


def estimated_metric_contributions(metric: Any, data: pd.DataFrame, problem_type: Any) -> Dict[str, np.ndarray]:
    """Calculate per-row contributions for a confidence-based performance estimate."""
    problem_type_value = getattr(problem_type, 'value', problem_type)
    if problem_type_value == 'classification_binary':
        return _binary_cbpe_contributions(metric, data)
    if problem_type_value == 'classification_multiclass':
        return _multiclass_cbpe_contributions(metric, data)
    raise InvalidArgumentsException(f"unsupported problem type '{problem_type_value}'.")
