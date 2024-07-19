#  Author:   Niels Nuyttens  <niels@nannyml.com>
#            Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

"""Calculates the data reconstruction error on unseen analysis data after fitting on reference data.

This calculator wraps a PCA transformation. It will be fitted on reference data when the `fit` method is called.
On calling the `calculate` method it will perform the inverse transformation on the analysis data and calculate
the euclidian distance between the analysis data and the reconstructed version of it.

This is the data reconstruction error, and it can be used as a measure of drift between
the reference and analysis data sets.

"""

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from flaml import AutoML
from lightgbm import LGBMClassifier
from pandas import MultiIndex
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

from nannyml.base import AbstractCalculator, _list_missing, _split_features_by_type
from nannyml.chunk import Chunker, Chunk
from nannyml.drift.multivariate.domain_classifier.result import Result
from nannyml.exceptions import InvalidArgumentsException

# from nannyml.sampling_error import SAMPLING_ERROR_RANGE
from nannyml.thresholds import ConstantThreshold, Threshold, calculate_threshold_values
from nannyml.usage_logging import UsageEvent, log_usage

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
    'random_state': 13,
    'reg_alpha': 0.0,
    'reg_lambda': 0.0,
    'silent': 'warn',
    'subsample': 1.0,
    'subsample_for_bin': 200000,
    'subsample_freq': 0,
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


class DomainClassifierCalculator(AbstractCalculator):
    """DomainClassifierCalculator implementation.

    Uses Drift Detection Classifier's cross validated performance as a measure of drift.
    """

    def __init__(
        self,
        feature_column_names: Union[str, List[str]],
        treat_as_categorical: Optional[Union[str, List[str]]] = None,
        timestamp_column_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_number: Optional[int] = None,
        chunk_period: Optional[str] = None,
        chunker: Optional[Chunker] = None,
        cv_folds_num: Optional[int] = 5,
        hyperparameters: Optional[Dict[str, Any]] = DEFAULT_LGBM_HYPERPARAMS,
        tune_hyperparameters: bool = False,
        hyperparameter_tuning_config: Optional[Dict[str, Any]] = DEFAULT_LGBM_HYPERPARAM_TUNING_CONFIG,
        threshold: Threshold = ConstantThreshold(lower=0.45, upper=0.65),
    ):
        """Create a new DomainClassifierCalculator instance.

        Parameters:
        -----------
        feature_column_names: List[str]
            A list containing the names of features in the provided data set. All of these features will be used by
            the multivariate classifier for drift detection to calculate an aggregate drift metric.
        treat_as_categorical: Optional[Union[str, List[str]]], default=None
            A list containing the names of features in the provided data set that should be treated as categorical.
            Needs not be exhaustive.
        timestamp_column_name:  Optional[str], default=None
            The name of the column containing the timestamp of the model prediction.
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
        cv_folds_num: Optional[int]
            Number of cross-validation folds to use when calculating DC discrimination value.
        hyperparameters : Dict[str, Any], default = None
            A dictionary used to provide your own custom hyperparameters when training the discrimination model.
            Check out the available hyperparameter options in the
            `LightGBM docs <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html>`_.
        tune_hyperparameters : bool, default = False
            A boolean controlling whether hypertuning should be performed on the internal regressor models
            whilst fitting on reference data.
            Tuning hyperparameters takes some time and does not guarantee better results,
            hence it defaults to `False`.
        threshold: Threshold, default=ConstantThreshold
            The threshold you wish to evaluate values on. Defaults to a ConstantThreshold with lower value
            of 0.45 and uppper value of 0.65.
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

        Example:
        --------
        >>> import nannyml as nml
        >>> # Load synthetic data
        >>> reference_df, analysis_df, _ = nml.load_synthetic_car_loan_dataset()
        >>> # Define feature columns
        >>> feature_column_names = [
        ...     col for col in reference_df.columns
        ...     if col not in non_feature_columns
        >>> ]
        >>> calc = nml.DomainClassifierCalculator(
        ...     feature_column_names=feature_column_names,
        ...     timestamp_column_name='timestamp',
        ...     chunk_size=5000
        >>> )
        >>> calc.fit(reference_df)
        >>> results = calc.calculate(analysis_df)
        >>> figure = results.plot()
        >>> figure.show()
        """
        super(DomainClassifierCalculator, self).__init__(
            chunk_size, chunk_number, chunk_period, chunker, timestamp_column_name
        )
        if isinstance(feature_column_names, str):
            feature_column_names = [feature_column_names]
        self.feature_column_names = feature_column_names

        if not treat_as_categorical:
            treat_as_categorical = []
        if isinstance(treat_as_categorical, str):
            treat_as_categorical = [treat_as_categorical]
        self.treat_as_categorical = treat_as_categorical

        self.continuous_column_names: List[str] = []
        self.categorical_column_names: List[str] = []

        self.hyperparameters = hyperparameters
        self.tune_hyperparameters = tune_hyperparameters
        self.hyperparameter_tuning_config = hyperparameter_tuning_config
        self.cv_folds_num = cv_folds_num

        self.threshold = threshold

        self.upper_threshold_value: Optional[float]
        self.lower_threshold_value: Optional[float]
        self._upper_threshold_value_limit: float = 1
        self._lower_threshold_value_limit: float = 0

        # # sampling error
        # self._sampling_error_components: Tuple = ()
        self.result: Optional[Result] = None
        self._is_fitted: bool = False

    @log_usage(UsageEvent.DC_CALC_FIT)
    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs):
        """Fits the DC calculator to a set of reference data."""
        if reference_data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing(self.feature_column_names, reference_data)

        self.continuous_column_names, self.categorical_column_names = _split_features_by_type(
            reference_data, self.feature_column_names
        )

        for column_name in self.treat_as_categorical:
            if column_name not in self.feature_column_names:
                self._logger.info(
                    f"ignoring 'treat_as_categorical' value '{column_name}' because it was not in "
                    f"listed column names"
                )
                break
            if column_name in self.continuous_column_names:
                self.continuous_column_names.remove(column_name)
            if column_name not in self.categorical_column_names:
                self.categorical_column_names.append(column_name)

        # Get timestamp column from chunker incase the calculator is initialized with a chunker without directly
        # been provided the timestamp column name.
        #
        # The reference data will be sorted according to the timestamp column (when available) to mimic
        # Chunker behavior. This means the reference data will be "aligned" with chunked reference data.
        # This way we can use chunk indices on the internal reference data copy.
        if self.chunker.timestamp_column_name:
            if self.chunker.timestamp_column_name not in list(reference_data.columns):
                raise InvalidArgumentsException(
                    f"timestamp column '{self.chunker.timestamp_column_name}' not in columns: {list(reference_data.columns)}."  # noqa: E501
                )
            self._reference_X = reference_data.sort_values(by=[self.chunker.timestamp_column_name]).reset_index(
                drop=True
            )[self.feature_column_names]
        else:
            self._reference_X = reference_data[self.feature_column_names]

        self.result = self._calculate(data=reference_data)
        self.result.data[('chunk', 'period')] = 'reference'

        self._is_fitted = True

        return self

    @log_usage(UsageEvent.DC_CALC_RUN)
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> Result:
        """Calculate the data DC calculator metric for a given data set."""
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing(self.feature_column_names, data)
        chunks = self.chunker.split(data, columns=self.feature_column_names)

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
                    # 'sampling_error': sampling_error(self._sampling_error_components, chunk.data),
                    'classifier_auroc_value': self._calculate_chunk(chunk=chunk),
                }
                for chunk in chunks
            ]
        )

        multilevel_index = _create_multilevel_index()
        res.columns = multilevel_index
        res = res.reset_index(drop=True)

        if not self._is_fitted:
            self._set_metric_thresholds(res)
            res = self._populate_alert_thresholds(res)
            self.result = Result(
                results_data=res,
                timestamp_column_name=self.timestamp_column_name,
                column_names=self.feature_column_names,
                categorical_column_names=self.categorical_column_names,
                continuous_column_names=self.continuous_column_names,
            )
        else:
            res = self._populate_alert_thresholds(res)
            self.result = self.result.filter(period='reference')  # type: ignore
            self.result.data = pd.concat([self.result.data, res], ignore_index=True)
        return self.result

    def _calculate_chunk(self, chunk: Chunk):
        if self._is_fitted:
            chunk_X = chunk.data[self.feature_column_names]
            reference_X = self._reference_X
            chunk_y = np.ones(len(chunk_X))
            reference_y = np.zeros(len(reference_X))
            X = pd.concat([reference_X, chunk_X], ignore_index=True)
            y = np.concatenate([reference_y, chunk_y])
        else:
            # Use information from chunk indices to identify reference chunk's location. This is possible because
            # both the internal reference data copy and the chunk data were sorted by timestamp, so these
            # indices align. This way we eliminate the need to combine these two data frames and drop duplicate rows,
            # which is a costly operation.
            X = self._reference_X
            y = np.zeros(len(X))
            y[chunk.start_index : chunk.end_index + 1] = 1

        df_X_transformed = preprocess_categorical_features(
            X, self.continuous_column_names, self.categorical_column_names
        )

        if self.tune_hyperparameters:
            self.tune_hyperparams(df_X_transformed, y)

        skf = StratifiedKFold(n_splits=self.cv_folds_num)
        all_preds = []
        all_tgts = []
        for i, (train_index, test_index) in enumerate(skf.split(df_X_transformed, y)):
            _trx = df_X_transformed.iloc[train_index]
            _try = y[train_index]
            _tsx = df_X_transformed.iloc[test_index]
            _tsy = y[test_index]
            with warnings.catch_warnings():
                # Ingore lightgbm's UserWarning: Using categorical_feature in Dataset.
                # We explicitly use that feature, don't spam the user
                warnings.filterwarnings("ignore", message="Using categorical_feature in Dataset.")
                model = LGBMClassifier(**self.hyperparameters)
                model.fit(_trx, _try, categorical_feature=self.categorical_column_names)
            preds = model.predict_proba(_tsx)[:, 1]
            all_preds.append(preds)
            all_tgts.append(_tsy)

        np_all_preds = np.concatenate(all_preds, axis=0)
        np_all_tgts = np.concatenate(all_tgts, axis=0)
        try:
            # catch case where all rows are duplicates
            result = roc_auc_score(np_all_tgts, np_all_preds)
        except ValueError as err:
            if str(err) != "Only one class present in y_true. ROC AUC score is not defined in that case.":
                raise
            else:
                # by definition if reference and chunk exactly match we can't discriminate
                result = 0.5
        return result

    def _set_metric_thresholds(self, result_data: pd.DataFrame):
        self.lower_threshold_value, self.upper_threshold_value = calculate_threshold_values(
            threshold=self.threshold,
            data=result_data.loc[:, ('domain_classifier_auroc', 'value')],
            lower_threshold_value_limit=self._lower_threshold_value_limit,
            upper_threshold_value_limit=self._upper_threshold_value_limit,
            logger=self._logger,
        )

    def _populate_alert_thresholds(self, result_data: pd.DataFrame) -> pd.DataFrame:
        result_data[('domain_classifier_auroc', 'upper_threshold')] = self.upper_threshold_value
        result_data[('domain_classifier_auroc', 'lower_threshold')] = self.lower_threshold_value
        result_data[('domain_classifier_auroc', 'alert')] = result_data.apply(
            lambda row: True
            if (
                row[('domain_classifier_auroc', 'value')] > row[('domain_classifier_auroc', 'upper_threshold')]
                or row[('domain_classifier_auroc', 'value')] < row[('domain_classifier_auroc', 'lower_threshold')]
            )
            else False,
            axis=1,
        )
        return result_data

    def tune_hyperparams(self, X: pd.DataFrame, y: np.ndarray):
        """Train an LGBM model while also performing hyperparameter tuning."""
        with warnings.catch_warnings():
            # Ingore lightgbm's UserWarning: Using categorical_feature in Dataset.
            # We explicitly use that feature, don't spam the user
            warnings.filterwarnings("ignore", message="Using categorical_feature in Dataset.")
            automl = AutoML()
            # TODO: Using categorical_feature
            automl.fit(
                X,
                y,
                **self.hyperparameter_tuning_config,
                categorical_feature=self.categorical_column_names,
            )
            self.hyperparameters = {**automl.model.estimator.get_params()}


def preprocess_categorical_features(
    X: pd.DataFrame, continuous_column_names: List[str], categorical_column_names: List[str]
) -> pd.DataFrame:
    """Preprodess categorical features."""
    X_cont = X[continuous_column_names]

    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_cat = pd.DataFrame({col_name: enc.fit_transform(X[[col_name]]).ravel() for col_name in categorical_column_names})

    return pd.concat([X_cat, X_cont], axis=1)


def _create_multilevel_index(include_thresholds: bool = False):
    chunk_column_names = ['key', 'chunk_index', 'start_index', 'end_index', 'start_date', 'end_date', 'period']
    results_column_names = [
        # 'sampling_error',
        'value',
        # 'upper_confidence_boundary',
        # 'lower_confidence_boundary',
    ]
    if include_thresholds:
        results_column_names += [
            'upper_threshold',
            'lower_threshold',
            'alert',
        ]
    chunk_tuples = [('chunk', chunk_column_name) for chunk_column_name in chunk_column_names]
    reconstruction_tuples = [('domain_classifier_auroc', column_name) for column_name in results_column_names]

    tuples = chunk_tuples + reconstruction_tuples

    return MultiIndex.from_tuples(tuples)
