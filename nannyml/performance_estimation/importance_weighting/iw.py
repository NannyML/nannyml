#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

"""A module with the implementation of the CBPE estimator.

The estimator manages a list of :class:`~nannyml.performance_estimation.confidence_based.metrics.Metric` instances,
constructed using the :class:`~nannyml.performance_estimation.confidence_based.metrics.MetricFactory`.

The estimator is then responsible for delegating the `fit` and `estimate` method calls to each of the managed
:class:`~nannyml.performance_estimation.confidence_based.metrics.Metric` instances and building a
:class:`~nannyml.performance_estimation.confidence_based.results.Result` object.

For more information, check out the `tutorial`_ and the `deep dive`_.

.. _tutorial:
    https://nannyml.readthedocs.io/en/stable/tutorials/performance_estimation/binary_performance_estimation.html

.. _deep dive:
    https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html#confidence-based-performance-estimation-cbpe
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import warnings

import numpy as np
import pandas as pd
from pandas import MultiIndex
from flaml import AutoML
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OrdinalEncoder

from nannyml._typing import ModelOutputsType, ProblemType, model_output_column_names
from nannyml.base import AbstractEstimator, _list_missing, _split_features_by_type
from nannyml.chunk import Chunk, Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_estimation.importance_weighting import SUPPORTED_METRIC_VALUES
from nannyml.performance_estimation.importance_weighting.metrics import MetricFactory
from nannyml.performance_estimation.importance_weighting.results import Result
from nannyml.thresholds import StandardDeviationThreshold, Threshold, calculate_threshold_values
from nannyml.usage_logging import UsageEvent, log_usage

DEFAULT_THRESHOLDS: Dict[str, Threshold] = {
    'roc_auc': StandardDeviationThreshold(),
    'f1': StandardDeviationThreshold(),
    'precision': StandardDeviationThreshold(),
    'recall': StandardDeviationThreshold(),
    'specificity': StandardDeviationThreshold(),
    'accuracy': StandardDeviationThreshold(),
    'confusion_matrix': StandardDeviationThreshold(),
    'business_value': StandardDeviationThreshold(),
}

DEFAULT_LGBM_HYPERPARAMS = {
    'boosting_type': 'gbdt',
    'class_weight': None,
    'colsample_bytree': 1.0,
    'importance_type': 'split',
    'learning_rate': 0.1,
    'max_depth': -1,
    'min_child_samples': 20,
    'min_child_weight': 0.001,
    'min_split_gain': 0.0,
    'n_estimators': 100,
    'n_jobs': -1,
    'num_leaves': 31,
    'objective': None,
    'random_state': 16,
    'reg_alpha': 0.0,
    'reg_lambda': 0.0,
    'silent': 'warn',
    'subsample': 1.0,
    'subsample_for_bin': 200000,
    'subsample_freq': 0
}


DEFAULT_LGBM_HYPERPARAM_TUNING_CONFIG = {
    "time_budget": 120,
    "metric": "roc_auc",
    "estimator_list": ['lgbm'],
    "eval_method": "cv",
    "hpo_method": "cfo",
    "n_splits": 5,
    "task": 'binary',
    "seed": 1,
    "verbose": 0,
}


def _default_encoder():
    return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)


class IW(AbstractEstimator):
    """Performance estimator using the Importance Weighting (IW) technique.

    Importance weigthing uses density ratio estimation between reference and chunk data.
    The density ratios are used to perform a weighted calculation of performance on reference data
    to estimate the performance on the chunk data.

    For more information, check out the `tutorial for binary classification`_,
    the `tutorial for multiclass classification`_ or the `deep dive`_.

    .. _tutorial for binary classification:
        https://nannyml.readthedocs.io/en/stable/tutorials/performance_estimation/binary_performance_estimation.html

    .. _tutorial for multiclass classification:
        https://nannyml.readthedocs.io/en/stable/tutorials/performance_estimation/multiclass_performance_estimation.html

    .. _deep dive:
        https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html#confidence-based-performance-estimation-cbpe
    """

    def __init__(
        self,
        metrics: Union[str, List[str]],
        feature_column_names: List[str],
        y_pred: str,
        y_pred_proba: ModelOutputsType,
        y_true: str,
        problem_type: Union[str, ProblemType],
        timestamp_column_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_number: Optional[int] = None,
        chunk_period: Optional[str] = None,
        chunker: Optional[Chunker] = None,
        hyperparameters: Optional[Dict[str, Any]] = DEFAULT_LGBM_HYPERPARAMS,
        tune_hyperparameters: bool = False,
        hyperparameter_tuning_config: Optional[Dict[str, Any]] = None,
        thresholds: Optional[Dict[str, Threshold]] = None,
        normalize_confusion_matrix: Optional[str] = None,
        business_value_matrix: Optional[Union[List, np.ndarray]] = None,
        normalize_business_value: Optional[str] = None,
        density_ratio_minimum_denominator: float = 0.05,
        density_ratio_minimum_value: float = 0.001
    ):
        """Initializes a new IW performance estimator.

        Parameters
        ----------
        feature_column_names: List[str]
            A list containing the names of features in the provided data set. All of these features will be used by
            the importance weighting calculator.
        y_true: str
            The name of the column containing target values (that are provided in reference data during fitting).
        y_pred_proba: Union[str, Dict[str, str]]
            Name(s) of the column(s) containing your model output.

                - For binary classification, pass a single string refering to the model output column.
                - For multiclass classification, pass a dictionary that maps a class string to the column name
                  model outputs for that class.
        y_pred: str
            The name of the column containing your model predictions.
        timestamp_column_name: str, default=None
            The name of the column containing the timestamp of the model prediction.
            If not given, plots will not use a time-based x-axis but will use the index of the chunks instead.
        metrics: Union[str, List[str]]
            A metric or list of metrics to calculate.

            Supported metrics by IW:

                - `roc_auc`
                - `f1`
                - `precision`
                - `recall`
                - `specificity`
                - `accuracy`
                - `confusion_matrix` - only for binary classification tasks
                - `business_value` - only for binary classification tasks
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
        hyperparameters : Dict[str, Any], default = None
            A dictionary used to provide your own custom hyperparameters when training the discrimination model.
            Check out the available hyperparameter options in the
            `LightGBM docs <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html>`_.
        tune_hyperparameters : bool, default = False
            A boolean controlling whether hypertuning should be performed on the internal regressor models
            whilst fitting on reference data.
            Tuning hyperparameters takes some time and does not guarantee better results,
            hence it defaults to `False`.
        hyperparameter_tuning_config : Dict[str, Any], default = None
            A dictionary that allows you to provide a custom hyperparameter tuning configuration when
            `tune_hyperparameters` has been set to `True`.
            The following dictionary is the default tuning configuration. It can be used as a template to modify::

                {
                    "time_budget": 15,
                    "metric": "mse",
                    "estimator_list": ['lgbm'],
                    "eval_method": "cv",
                    "hpo_method": "cfo",
                    "n_splits": 5,
                    "task": 'regression',
                    "seed": 1,
                    "verbose": 0,
                }

            For an overview of possible parameters for the tuning process check out the
            `FLAML documentation <https://microsoft.github.io/FLAML/docs/reference/automl/automl>`_.
        thresholds: dict
            The default values are::

                {
                    'roc_auc': StandardDeviationThreshold(),
                    'f1': StandardDeviationThreshold(),
                    'precision': StandardDeviationThreshold(),
                    'recall': StandardDeviationThreshold(),
                    'specificity': StandardDeviationThreshold(),
                    'accuracy': StandardDeviationThreshold(),
                    'confusion_matrix': StandardDeviationThreshold(),  # only for binary classification
                    'business_value': StandardDeviationThreshold(),  # only for binary classification
                }

            A dictionary allowing users to set a custom threshold for each method. It links a `Threshold` subclass
            to a method name. This dictionary is optional.
            When a dictionary is given its values will override the default values. If no dictionary is given a default
            will be applied.
        problem_type: Union[str, ProblemType]
            Determines which CBPE implementation to use. Allowed problem type values are 'classification_binary' and
            'classification_multiclass'.
        normalize_confusion_matrix: str, default=None
            Determines how the confusion matrix will be normalized. Allowed values are None, 'all', 'true' and
            'predicted'.

                - None - the confusion matrix will not be normalized and the counts for each cell of the matrix \
                will be returned.
                - 'all' - the confusion matrix will be normalized by the total number of observations.
                - 'true' - the confusion matrix will be normalized by the total number of observations for each true  \
                class.
                - 'predicted' - the confusion matrix will be normalized by the total number of observations for each \
                predicted class.
        business_value_matrix: Optional[Union[List, np.ndarray]], default=None
            A 2x2 matrix that specifies the value of each cell in the confusion matrix.
            The format of the business value matrix must be specified as [[value_of_TN, value_of_FP], \
            [value_of_FN, value_of_TP]]. Required when estimating the 'business_value' metric.
        normalize_business_value: str, default=None
            Determines how the business value will be normalized. Allowed values are None and
            'per_prediction'.

            - None - the business value will not be normalized and the value returned will be the total value per chunk.
            - 'per_prediction' - the value will be normalized by the number of predictions in the chunk.
        density_ratio_minimum_denominator: float, default=0.05,
            When calculating density ratio limit the minimum value of the denominator. This introduces a solf limit
            how big the density ratio can be.
        density_ratio_minimum_value: float, default=0.001
            When calculating density ratio limit the minimum value of the density ratio. We don't want data
            to be completely ignored because it can cause problems. 

        Examples
        --------
        Using CBPE to estimate the perfomance of a model for a binary classification problem.

        >>> import nannyml as nml
        >>> from IPython.display import display
        >>> reference_df = nml.load_synthetic_car_loan_dataset()[0]
        >>> analysis_df = nml.load_synthetic_car_loan_dataset()[1]
        >>> display(reference_df.head(3))
        >>> estimator = nml.CBPE(
        ...     y_pred_proba='y_pred_proba',
        ...     y_pred='y_pred',
        ...     y_true='repaid',
        ...     timestamp_column_name='timestamp',
        ...     metrics=['roc_auc', 'accuracy', 'f1'],
        ...     chunk_size=5000,
        ...     problem_type='classification_binary',
        >>> )
        >>> estimator.fit(reference_df)
        >>> results = estimator.estimate(analysis_df)
        >>> display(results.filter(period='analysis').to_df())
        >>> metric_fig = results.plot()
        >>> metric_fig.show()

        Using CBPE to estimate the perfomance of a model for a multiclass classification problem.

        >>> import nannyml as nml
        >>> reference_df, analysis_df, _ = nml.load_synthetic_multiclass_classification_dataset()
        >>> estimator = nml.CBPE(
        ...     y_pred_proba={
        ...         'prepaid_card': 'y_pred_proba_prepaid_card',
        ...         'highstreet_card': 'y_pred_proba_highstreet_card',
        ...         'upmarket_card': 'y_pred_proba_upmarket_card'},
        ...     y_pred='y_pred',
        ...     y_true='y_true',
        ...     timestamp_column_name='timestamp',
        ...     problem_type='classification_multiclass',
        ...     metrics=['roc_auc', 'f1'],
        ...     chunk_size=6000,
        >>> )
        >>> estimator.fit(reference_df)
        >>> results = estimator.estimate(analysis_df)
        >>> metric_fig = results.plot()
        >>> metric_fig.show()
        """
        super().__init__(chunk_size, chunk_number, chunk_period, chunker, timestamp_column_name)

        self.feature_column_names = feature_column_names
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba

        if metrics is None or len(metrics) == 0:
            raise InvalidArgumentsException(
                "no metrics provided. Please provide a non-empty list of metrics."
                f"Supported values are {SUPPORTED_METRIC_VALUES}."
            )

        valid_normalizations = [None, 'all', 'pred', 'true']
        if normalize_confusion_matrix not in valid_normalizations:
            raise InvalidArgumentsException(
                f"'normalize_confusion_matrix' given was '{normalize_confusion_matrix}'. "
                f"Binary use cases require 'normalize_confusion_matrix' to be one of {valid_normalizations}."
            )

        if normalize_business_value not in [None, "per_prediction"]:
            raise InvalidArgumentsException(
                f"normalize_business_value must be None or 'per_prediction', but got '{normalize_business_value}'"
            )

        if isinstance(problem_type, str):
            self.problem_type = ProblemType.parse(problem_type)
        else:
            self.problem_type = problem_type

        if hyperparameter_tuning_config is None:
            hyperparameter_tuning_config = DEFAULT_LGBM_HYPERPARAM_TUNING_CONFIG
        self.hyperparameter_tuning_config = hyperparameter_tuning_config
        self.tune_hyperparameters = tune_hyperparameters
        self.hyperparameters = hyperparameters

        self.thresholds = DEFAULT_THRESHOLDS
        if thresholds:
            self.thresholds.update(**thresholds)

        self.density_ratio_minimum_denominator = density_ratio_minimum_denominator
        self.density_ratio_minimum_value = density_ratio_minimum_value

        if isinstance(metrics, str):
            metrics = [metrics]

        self.metrics = []
        for metric in metrics:
            if metric not in SUPPORTED_METRIC_VALUES:
                raise InvalidArgumentsException(
                    f"unknown metric key '{metric}' given. " f"Should be one of {SUPPORTED_METRIC_VALUES}."
                )
            self.metrics.append(
                MetricFactory.create(
                    metric,
                    self.problem_type,
                    y_pred_proba=self.y_pred_proba,
                    y_pred=self.y_pred,
                    y_true=self.y_true,
                    timestamp_column_name=self.timestamp_column_name,
                    chunker=self.chunker,
                    threshold=self.thresholds[metric],
                    normalize_confusion_matrix=normalize_confusion_matrix,
                    business_value_matrix=business_value_matrix,
                    normalize_business_value=normalize_business_value,
                )
            )

        self.result: Optional[Result] = None
        self._categorical_encoders: defaultdict = defaultdict(_default_encoder)

    @log_usage(UsageEvent.IW_ESTIMATOR_FIT, metadata_from_self=['metrics', 'problem_type'])
    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> CBPE:
        """Fits the drift calculator using a set of reference data.

        Parameters
        ----------
        reference_data : pd.DataFrame
            A reference data set containing predictions (labels and/or probabilities) and target values.

        Returns
        -------
        estimator: PerformanceEstimator
            The fitted estimator.
        """
        if reference_data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')
        _list_missing(
            [self.y_true, self.y_pred] + self.feature_column_names + model_output_column_names(self.y_pred_proba),
            reference_data
        )
        self.continuous_column_names, self.categorical_column_names = _split_features_by_type(
            reference_data, self.feature_column_names
        )
        # TODO: Need a better way of doing this for big data.
        self.reference_data = reference_data[
            [self.y_true, self.y_pred] + self.feature_column_names + model_output_column_names(self.y_pred_proba)
        ]
        for metric in self.metrics:
            metric.fit(reference_data)
        self.result = self._estimate(reference_data)
        assert self.result
        self.result.data[('chunk', 'period')] = 'reference'     
        return self

    @log_usage(UsageEvent.IW_ESTIMATOR_RUN, metadata_from_self=['metrics', 'problem_type'])
    def _estimate(self, data: pd.DataFrame, *args, **kwargs) -> Result:
        """Calculates the data reconstruction drift for a given data set.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to calculate the reconstruction drift for.

        Returns
        -------
        estimates: PerformanceEstimatorResult
            A :class:`result<nannyml.performance_estimation.confidence_based.results.Result>`
            object where each row represents a :class:`~nannyml.chunk.Chunk`,
            containing :class:`~nannyml.chunk.Chunk` properties and the estimated metrics
            for that :class:`~nannyml.chunk.Chunk`.
        """
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')
        
        _list_missing(
            [self.y_pred] + self.feature_column_names + model_output_column_names(self.y_pred_proba),
            data
        )

        chunks = self.chunker.split(data)

        res = pd.DataFrame.from_records(
            [
                {
                    'key': chunk.key,
                    'chunk_index': chunk.chunk_index,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'start_date': chunk.start_datetime,
                    'end_date': chunk.end_datetime,
                    'period': 'analysis',
                    **self._estimate_chunk(chunk),
                }
                for chunk in chunks
            ]
        )

        metric_column_names = [name for metric in self.metrics for name in metric.column_names]
        multilevel_index = _create_multilevel_index(metric_names=metric_column_names)

        res.columns = multilevel_index
        res = res.reset_index(drop=True)

        if self.result is None:
            self.metrics = self._set_metric_thresholds(res)
            res = self._populate_alert_thresholds(res)
            self.result = Result(
                results_data=res,
                y_pred_proba=self.y_pred_proba,
                y_pred=self.y_pred,
                y_true=self.y_true,
                timestamp_column_name=self.timestamp_column_name,
                metrics=self.metrics,
                chunker=self.chunker,
                problem_type=self.problem_type,
            )
        else:
            res = self._populate_alert_thresholds(res)
            self.result = self.result.filter(period='reference')
            self.result.data = pd.concat([self.result.data, res]).reset_index(drop=True)

        return self.result
    
    def _preprocess_data_for_dre_model(self, reference_X: pd.DataFrame, chunk_X: pd.DataFrame):
        """Preprocess Data for dre model.

        Parameters:
        -----------
        reference_X: pd.DataFrame
            Pandas dataframe containing only feature column names from reference data.
        chunk_X: pd.DataFrame
            Pandas dataframe containing only feature column names from chunk data.
        
        Returns:
        _X: pd.DataFrame
            training dataframe for dre model
        _y: pd.DataFrame
            label dataframe for dre model
        """
        chunk_y = np.ones(len(chunk_X))
        reference_y = np.zeros(len(reference_X))

        dfx = pd.concat([reference_X, chunk_X]).reset_index(drop=True)
        npy = np.concatenate([reference_y, chunk_y])

        # drop duplicate columns
        dfx['__target__'] = npy
        dfx = dfx.drop_duplicates(subset=self.feature_column_names, keep='last').reset_index(drop=True)
        _y = dfx['__target__']
        dfx.drop('__target__', axis=1, inplace=True)

        dfx_cont = dfx[self.continuous_column_names]
        dfx_cat = pd.DataFrame({
            col_name: self._categorical_encoders[col_name].fit_transform(dfx[[col_name]]).ravel() for col_name in self.categorical_column_names
        })
        _x = pd.concat([dfx_cat, dfx_cont], axis=1)
        _x = _x[self.categorical_column_names + self.continuous_column_names]
        return _x, _y

    def _preprocess_ref_for_pred_proba(self, reference_data: pd.DataFrame):
        for col_name in self.categorical_column_names:
            reference_data[col_name] = self._categorical_encoders[col_name].transform(reference_data[[col_name]]).ravel()
        # order of columns must be preserved between training and predictions.
        reference_data = reference_data[self.categorical_column_names + self.continuous_column_names]
        return reference_data

    def _train_dre_model(self, _x, _y):
        self._logger.debug("Started training direct ratio estimation model for chunk.")
        # Ingore lightgbm's UserWarning: Using categorical_feature in Dataset.
        # We explicitly use that feature, don't spam the user
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Using categorical_feature in Dataset.")
            if self.hyperparameters:
                self._logger.debug("'hyperparameters' set: using custom hyperparameters")
                self._logger.debug(f"'hyperparameters': {self.hyperparameters}")
                model = LGBMClassifier(**self.hyperparameters)
            elif self.tune_hyperparameters:
                self._logger.debug(
                    f"'tune_hyperparameters' set to '{self.tune_hyperparameters}': " f"performing hyperparameter tuning"
                )
                self._logger.debug(f'hyperparameter tuning configuration: {self.hyperparameter_tuning_config}')
                automl = AutoML()
                automl.fit(_x, _y, **self.hyperparameter_tuning_config, categorical_feature=self.categorical_column_names)
                model = LGBMClassifier(**automl.model.estimator.get_params())
            else:
                self._logger.debug(
                    f"'tune_hyperparameters' set to '{self.tune_hyperparameters}': skipping hyperparameter tuning"
                )
                self._logger.debug("'hyperparameters' not set: using default hyperparameters")
                model = LGBMClassifier()
            model.fit(_x, _y, categorical_feature=self.categorical_column_names)
        return model

    def _calculate_weights(self, weight_y_pred_probas :np.ndarray, size_chunk, size_reference) -> np.ndarray:
        correcting_factor = size_reference/size_chunk
        denominator = np.maximum(self.density_ratio_minimum_denominator, 1-weight_y_pred_probas)
        likelihood_ratio = correcting_factor*weight_y_pred_probas/denominator
        likelihood_ratio = np.maximum(likelihood_ratio, self.density_ratio_minimum_value)# avoid 0 weights
        return likelihood_ratio

    def _estimate_chunk(self, chunk: Chunk) -> Dict:

        # Train Ratio estimation model here and predict weights on ref data
        X, y = self._preprocess_data_for_dre_model(
            self.reference_data[self.feature_column_names].copy(deep=True),
            chunk.data[self.feature_column_names].copy(deep=True)
        )
        model = self._train_dre_model(X,y)
        ref_transformed = self._preprocess_ref_for_pred_proba(
            self.reference_data[self.feature_column_names].copy(deep=True)
        )
        reference_dre_probas = model.predict_proba(ref_transformed)[:, 1]
        # print("debug reference_dre_probas")
        # print(reference_dre_probas.mean())
        # print(reference_dre_probas.std())
        reference_weights = self._calculate_weights(
            reference_dre_probas,
            chunk.data.shape[0],
            ref_transformed.shape[0],
        )
        # print("debug weights")
        # print(reference_weights.mean())
        # print(reference_weights.std())

        _selected_output_colums = [self.y_true, self.y_pred] + model_output_column_names(self.y_pred_proba)

        chunk_records: Dict[str, Any] = {}
        for metric in self.metrics:
            chunk_record = metric.get_chunk_record(
                chunk.data[
                    # because y_true may not be present in chunk data
                    [col for col in _selected_output_colums if col in list(chunk.data.columns)]
                ].copy(deep=True),
                self.reference_data[_selected_output_colums].copy(deep=True),
                reference_weights
            )
            # add the chunk record to the chunk_records dict
            chunk_records.update(chunk_record)
        return chunk_records
    
    def _set_metric_thresholds(self, result_data: pd.DataFrame) -> List:
        updated_metrics = []
        for metric in self.metrics:
            if metric.name != "confusion_matrix":
                metric.lower_threshold_value, metric.upper_threshold_value = calculate_threshold_values(
                    threshold=metric.threshold,
                    data=result_data.loc[:, (metric.column_name, 'realized')],
                    lower_threshold_value_limit=metric.lower_threshold_value_limit,
                    upper_threshold_value_limit=metric.upper_threshold_value_limit,
                    logger=self._logger,
                    metric_name=metric.display_name,
                )
                updated_metrics.append(metric)
            else:
                if self.problem_type == ProblemType.CLASSIFICATION_BINARY:
                    metric.true_positive_lower_threshold, metric.true_positive_upper_threshold = calculate_threshold_values(
                        threshold=metric.threshold,
                        data=result_data.loc[:, ("true_positive", "realized")],
                        lower_threshold_value_limit=metric.lower_threshold_value_limit,
                        upper_threshold_value_limit=metric.upper_threshold_value_limit,
                        logger=self._logger,
                        metric_name="true_positive",  # component 0 // to iterate
                    )
                    metric.true_negative_lower_threshold, metric.true_negative_upper_threshold = calculate_threshold_values(
                        threshold=metric.threshold,
                        data=result_data.loc[:, ("true_negative", "realized")],
                        lower_threshold_value_limit=metric.lower_threshold_value_limit,
                        upper_threshold_value_limit=metric.upper_threshold_value_limit,
                        logger=self._logger,
                        metric_name="true_negative",  # component 1 // to iterate
                    )
                    (
                        metric.false_positive_lower_threshold,
                        metric.false_positive_upper_threshold,
                    ) = calculate_threshold_values(
                        threshold=metric.threshold,
                        data=result_data.loc[:, ("false_positive", "realized")],
                        lower_threshold_value_limit=metric.lower_threshold_value_limit,
                        upper_threshold_value_limit=metric.upper_threshold_value_limit,
                        logger=self._logger,
                        metric_name="false_positive",  # component 2 // to iterate
                    )
                    (
                        metric.false_negative_lower_threshold,
                        metric.false_negative_upper_threshold,
                    ) = calculate_threshold_values(
                        threshold=metric.threshold,
                        data=result_data.loc[:, ("false_negative", "realized")],
                        lower_threshold_value_limit=metric.lower_threshold_value_limit,
                        upper_threshold_value_limit=metric.upper_threshold_value_limit,
                        logger=self._logger,
                        metric_name="false_negative",  # component 3 // to iterate
                    )
                    updated_metrics.append(metric)
                elif self.problem_type == ProblemType.CLASSIFICATION_MULTICLASS:
                    alert_thresholds_dict = {}
                    num_classes = len(metric.classes)
                    for i in range(num_classes):
                        for j in range(num_classes):
                            lower_threshold_value, upper_threshold_value = calculate_threshold_values(
                                threshold=metric.threshold,
                                # data=realized_chunk_performance[:, i, j],
                                data=result_data.loc[:, (f"true_{metric.classes[i]}_pred_{metric.classes[j]}", "realized")],
                                lower_threshold_value_limit=metric.lower_threshold_value_limit,
                                upper_threshold_value_limit=metric.upper_threshold_value_limit,
                            )
                            alert_thresholds_dict[f"true_{metric.classes[i]}_pred_{metric.classes[j]}"] = (
                                lower_threshold_value,
                                upper_threshold_value,
                            )
                    metric.alert_thresholds_dict = alert_thresholds_dict
                    updated_metrics.append(metric)
        return updated_metrics

    def _populate_alert_thresholds(self, result_data: pd.DataFrame) -> pd.DataFrame:
        for metric in self.metrics:
            if metric.name != "confusion_matrix":
                column_name = metric.column_name
                result_data[(column_name, 'upper_threshold')] = metric.upper_threshold_value
                result_data[(column_name, 'lower_threshold')] = metric.lower_threshold_value
                result_data[(column_name, 'alert')] = result_data.apply(
                    lambda row: True
                    if (
                        row[(column_name, 'value')] > row[(column_name, 'upper_threshold')]
                        or row[(column_name, 'value')] < row[(column_name, 'lower_threshold')]
                    )
                    else False,
                    axis=1,
                )
                del column_name
            else:
                if self.problem_type == ProblemType.CLASSIFICATION_BINARY:
                    column_name = 'true_positive'
                    result_data[(column_name, 'upper_threshold')] = metric.true_positive_upper_threshold
                    result_data[(column_name, 'lower_threshold')] = metric.true_positive_lower_threshold
                    result_data[(column_name, 'alert')] = result_data.apply(
                        lambda row: True
                        if (
                            row[(column_name, 'value')] > row[(column_name, 'upper_threshold')]
                            or row[(column_name, 'value')] < row[(column_name, 'lower_threshold')]
                        )
                        else False,
                        axis=1,
                    )
                    column_name = 'true_negative'
                    result_data[(column_name, 'upper_threshold')] = metric.true_negative_upper_threshold
                    result_data[(column_name, 'lower_threshold')] = metric.true_negative_lower_threshold
                    result_data[(column_name, 'alert')] = result_data.apply(
                        lambda row: True
                        if (
                            row[(column_name, 'value')] > row[(column_name, 'upper_threshold')]
                            or row[(column_name, 'value')] < row[(column_name, 'lower_threshold')]
                        )
                        else False,
                        axis=1,
                    )
                    column_name = 'false_positive'
                    result_data[(column_name, 'upper_threshold')] = metric.false_positive_upper_threshold
                    result_data[(column_name, 'lower_threshold')] = metric.false_positive_lower_threshold
                    result_data[(column_name, 'alert')] = result_data.apply(
                        lambda row: True
                        if (
                            row[(column_name, 'value')] > row[(column_name, 'upper_threshold')]
                            or row[(column_name, 'value')] < row[(column_name, 'lower_threshold')]
                        )
                        else False,
                        axis=1,
                    )
                    column_name = 'false_negative'
                    result_data[(column_name, 'upper_threshold')] = metric.false_negative_upper_threshold
                    result_data[(column_name, 'lower_threshold')] = metric.false_negative_lower_threshold
                    result_data[(column_name, 'alert')] = result_data.apply(
                        lambda row: True
                        if (
                            row[(column_name, 'value')] > row[(column_name, 'upper_threshold')]
                            or row[(column_name, 'value')] < row[(column_name, 'lower_threshold')]
                        )
                        else False,
                        axis=1,
                    )
                else:
                    # column names are all elements of f"true_{self.classes[i]}_pred_{self.classes[j]}"
                    for column_name in metric.column_names:
                        _lower_threshold_value = metric.alert_thresholds_dict[column_name][0]
                        _upper_threshold_value = metric.alert_thresholds_dict[column_name][1]
                        result_data[(column_name, 'upper_threshold')] = _upper_threshold_value
                        result_data[(column_name, 'lower_threshold')] = _lower_threshold_value
                        result_data[(column_name, 'alert')] = result_data.apply(
                            lambda row: True
                            if (
                                row[(column_name, 'value')] > row[(column_name, 'upper_threshold')]
                                or row[(column_name, 'value')] < row[(column_name, 'lower_threshold')]
                            )
                            else False,
                            axis=1,
                        )
        return result_data


def _create_multilevel_index(metric_names: List[str], include_thresholds: bool = False) -> MultiIndex:
    chunk_column_names = [
        'key',
        'chunk_index',
        'start_index',
        'end_index',
        'start_date',
        'end_date',
        'period',
    ]
    results_column_names = [
        'value',
        'sampling_error',
        'realized',
        'upper_confidence_boundary',
        'lower_confidence_boundary',
    ]
    if include_thresholds:
        results_column_names += [
            'upper_threshold',
            'lower_threshold',
            'alert',
        ]
    chunk_tuples = [('chunk', chunk_column_name) for chunk_column_name in chunk_column_names]
    reconstruction_tuples = [
        (metric_name, column_name) for metric_name in metric_names for column_name in results_column_names
    ]
    tuples = chunk_tuples + reconstruction_tuples
    return MultiIndex.from_tuples(tuples)
