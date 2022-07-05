#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Statistical drift calculation using `Kolmogorov-Smirnov` and `chi2-contingency` tests."""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from nannyml._typing import ModelOutputsType, model_output_column_names
from nannyml.base import AbstractCalculator, _list_missing, _split_features_by_type
from nannyml.chunk import Chunker
from nannyml.drift.model_outputs.univariate.statistical.results import UnivariateDriftResult
from nannyml.exceptions import InvalidArgumentsException

ALERT_THRESHOLD_P_VALUE = 0.05


class StatisticalOutputDriftCalculator(AbstractCalculator):
    """A drift calculator that relies on statistics to detect drift."""

    def __init__(
        self,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        timestamp_column_name: str,
        chunk_size: int = None,
        chunk_number: int = None,
        chunk_period: str = None,
        chunker: Chunker = None,
    ):
        """Constructs a new StatisticalOutputDriftCalculator.

        Parameters
        ----------
        y_pred_proba: ModelOutputsType
            Name(s) of the column(s) containing your model output.
            Pass a single string when there is only a single model output column, e.g. in binary classification cases.
            Pass a dictionary when working with multiple output columns, e.g. in multiclass classification cases.
            The dictionary maps a class/label string to the column name containing model outputs for that class/label.
        y_pred: str
            The name of the column containing your model predictions.
        timestamp_column_name: str
            The name of the column containing the timestamp of the model prediction.
        chunk_size: int
            Splits the data into chunks containing `chunks_size` observations.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunk_number: int
            Splits the data into `chunk_number` pieces.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunk_period: str
            Splits the data according to the given period.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunker : Chunker
            The `Chunker` used to split the data sets into a lists of chunks.

        Examples
        --------
        >>> import nannyml as nml
        >>> ref_df, ana_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>> metadata = nml.extract_metadata(ref_df)
        >>> # Create a calculator that will chunk by week
        >>> drift_calc = nml.UnivariateStatisticalDriftCalculator(model_metadata=metadata, chunk_period='W')
        """
        super(StatisticalOutputDriftCalculator, self).__init__(chunk_size, chunk_number, chunk_period, chunker)

        self.y_pred_proba = y_pred_proba
        self.y_pred = y_pred
        self.timestamp_column_name = timestamp_column_name

        self.previous_reference_data: Optional[pd.DataFrame] = None
        self.previous_reference_results: Optional[pd.DataFrame] = None
        self.previous_analysis_data: Optional[pd.DataFrame] = None

    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs):
        """Fits the drift calculator using a set of reference data.

        Parameters
        ----------
        reference_data : pd.DataFrame
            A reference data set containing predictions (labels and/or probabilities) and target values.

        Returns
        -------
        calculator: StatisticalOutputDriftCalculator
            The fitted calculator.

        Examples
        --------
        >>> import nannyml as nml
        >>> ref_df, ana_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>> metadata = nml.extract_metadata(ref_df, model_type=nml.ModelType.CLASSIFICATION_BINARY)
        >>> # Create a calculator and fit it
        >>> drift_calc = nml.UnivariateStatisticalDriftCalculator(model_metadata=metadata, chunk_period='W').fit(ref_df)

        """
        if reference_data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing([self.y_pred] + model_output_column_names(self.y_pred_proba), reference_data)

        self.previous_reference_data = reference_data.copy()
        self.previous_reference_results = self._calculate(reference_data).data

        return self

    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> UnivariateDriftResult:
        """Calculates the data reconstruction drift for a given data set.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to calculate the reconstruction drift for.

        Returns
        -------
        reconstruction_drift: UnivariateDriftResult
            A :class:`result<nannyml.drift.model_inputs.univariate.statistical.results.
            UnivariateStatisticalDriftCalculatorResult>`
            object where each row represents a :class:`~nannyml.chunk.Chunk`,
            containing :class:`~nannyml.chunk.Chunk` properties and the reconstruction_drift calculated
            for that :class:`~nannyml.chunk.Chunk`.

        Examples
        --------
        >>> import nannyml as nml
        >>> ref_df, ana_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>> metadata = nml.extract_metadata(ref_df, model_type=nml.ModelType.CLASSIFICATION_BINARY)
        >>> # Create a calculator and fit it
        >>> drift_calc = nml.UnivariateStatisticalDriftCalculator(model_metadata=metadata, chunk_period='W').fit(ref_df)
        >>> drift = drift_calc.calculate(data)
        """
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing([self.y_pred] + model_output_column_names(self.y_pred_proba), data)

        columns = [self.y_pred]
        if isinstance(self.y_pred_proba, Dict):
            columns += [v for _, v in self.y_pred_proba.items()]
        elif isinstance(self.y_pred_proba, str):
            columns += [self.y_pred_proba]
        else:
            raise InvalidArgumentsException(
                "parameter 'y_pred_proba' is of type '{type(y_pred_proba)}' "
                "but should be of type 'Union[str, Dict[str, str].'"
            )

        continuous_columns, categorical_columns = _split_features_by_type(data, columns)

        chunks = self.chunker.split(
            data, columns=columns, minimum_chunk_size=500, timestamp_column_name=self.timestamp_column_name
        )

        chunk_drifts = []
        # Calculate chunk-wise drift statistics.
        # Append all into resulting DataFrame indexed by chunk key.
        for chunk in chunks:
            chunk_drift: Dict[str, Any] = {
                'key': chunk.key,
                'start_index': chunk.start_index,
                'end_index': chunk.end_index,
                'start_date': chunk.start_datetime,
                'end_date': chunk.end_datetime,
                'period': 'analysis' if chunk.is_transition else chunk.period,
            }

            for column in categorical_columns:
                statistic, p_value, _, _ = chi2_contingency(
                    pd.concat(
                        [
                            self.previous_reference_data[column].value_counts(),  # type: ignore
                            chunk.data[column].value_counts(),
                        ],
                        axis=1,
                    ).fillna(0)
                )
                chunk_drift[f'{column}_chi2'] = statistic
                chunk_drift[f'{column}_p_value'] = np.round(p_value, decimals=3)
                chunk_drift[f'{column}_alert'] = p_value < ALERT_THRESHOLD_P_VALUE
                chunk_drift[f'{column}_threshold'] = ALERT_THRESHOLD_P_VALUE

            for column in continuous_columns:
                statistic, p_value = ks_2samp(self.previous_reference_data[column], chunk.data[column])  # type: ignore
                chunk_drift[f'{column}_dstat'] = statistic
                chunk_drift[f'{column}_p_value'] = np.round(p_value, decimals=3)
                chunk_drift[f'{column}_alert'] = p_value < ALERT_THRESHOLD_P_VALUE
                chunk_drift[f'{column}_threshold'] = ALERT_THRESHOLD_P_VALUE

            chunk_drifts.append(chunk_drift)

        res = pd.DataFrame.from_records(chunk_drifts)
        res = res.reset_index(drop=True)

        self.previous_analysis_data = data.copy()

        return UnivariateDriftResult(results_data=res, calculator=self)
