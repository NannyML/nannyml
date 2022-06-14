#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Statistical drift calculation using `Kolmogorov-Smirnov` and `chi2-contingency` tests."""
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from nannyml.chunk import Chunker
from nannyml.drift.base import DriftCalculator
from nannyml.drift.model_outputs.univariate.statistical.results import UnivariateDriftResult
from nannyml.exceptions import CalculatorNotFittedException, MissingMetadataException
from nannyml.metadata import BinaryClassificationMetadata, MulticlassClassificationMetadata, RegressionMetadata
from nannyml.metadata.base import NML_METADATA_COLUMNS, NML_METADATA_PERIOD_COLUMN_NAME, ModelMetadata
from nannyml.preprocessing import preprocess

ALERT_THRESHOLD_P_VALUE = 0.05


class UnivariateStatisticalDriftCalculator(DriftCalculator):
    """A drift calculator that relies on statistics to detect drift."""

    def __init__(
        self,
        model_metadata: ModelMetadata,
        chunk_size: int = None,
        chunk_number: int = None,
        chunk_period: str = None,
        chunker: Chunker = None,
    ):
        """Constructs a new UnivariateStatisticalDriftCalculator.

        Parameters
        ----------
        model_metadata: ModelMetadata
            Metadata for the model whose data is to be processed.
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
        super(UnivariateStatisticalDriftCalculator, self).__init__(
            model_metadata, None, chunk_size, chunk_number, chunk_period, chunker
        )

        self._reference_data = None

    def fit(self, reference_data: pd.DataFrame):
        """Fits the drift calculator using a set of reference data.

        Parameters
        ----------
        reference_data : pd.DataFrame
            A reference data set containing predictions (labels and/or probabilities) and target values.

        Returns
        -------
        calculator: DriftCalculator
            The fitted calculator.

        Examples
        --------
        >>> import nannyml as nml
        >>> ref_df, ana_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>> metadata = nml.extract_metadata(ref_df, model_type=nml.ModelType.CLASSIFICATION_BINARY)
        >>> # Create a calculator and fit it
        >>> drift_calc = nml.UnivariateStatisticalDriftCalculator(model_metadata=metadata, chunk_period='W').fit(ref_df)

        """
        # check metadata for required properties
        self.model_metadata.check_has_fields(['period_column_name', 'timestamp_column_name', 'prediction_column_name'])

        reference_data = preprocess(data=reference_data, metadata=self.model_metadata, reference=True)

        # store state
        self._reference_data = reference_data.copy(deep=True)

        return self

    def calculate(
        self,
        data: pd.DataFrame,
    ) -> UnivariateDriftResult:
        """Calculates the data reconstruction drift for a given data set.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to calculate the reconstruction drift for.

        Returns
        -------
        reconstruction_drift: UnivariateDriftResult
            A :class:`result<nannyml.drift.model_inputs.univariate.statistical.results.UnivariateDriftResult>`
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
        # Check metadata for required properties
        self.model_metadata.check_has_fields(['period_column_name', 'timestamp_column_name', 'prediction_column_name'])
        prediction_column_names, predicted_probabilities_column_names = _get_predictions_and_scores(self.model_metadata)
        data = preprocess(data=data, metadata=self.model_metadata)

        features_and_metadata = NML_METADATA_COLUMNS + prediction_column_names + predicted_probabilities_column_names
        chunks = self.chunker.split(data, columns=features_and_metadata, minimum_chunk_size=500)

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

            present_continuous_column_names = list(
                set(chunk.data.columns) & set(prediction_column_names + predicted_probabilities_column_names)
            )
            for column in present_continuous_column_names:
                statistic, p_value = ks_2samp(self._reference_data[column], chunk.data[column])  # type: ignore
                chunk_drift[f'{column}_dstat'] = statistic
                chunk_drift[f'{column}_p_value'] = np.round(p_value, decimals=3)
                chunk_drift[f'{column}_alert'] = (p_value < ALERT_THRESHOLD_P_VALUE) and (
                    chunk.data[NML_METADATA_PERIOD_COLUMN_NAME] == 'analysis'
                ).all()
                chunk_drift[f'{column}_threshold'] = ALERT_THRESHOLD_P_VALUE

            chunk_drifts.append(chunk_drift)

        res = pd.DataFrame.from_records(chunk_drifts)
        res = res.reset_index(drop=True)
        res.attrs['nml_drift_calculator'] = __name__

        if self.chunker is None:
            raise CalculatorNotFittedException(
                'chunker has not been set. '
                'Please ensure you run ``calculator.fit()`` '
                'before running ``calculator.calculate()``'
            )

        return UnivariateDriftResult(analysis_data=chunks, drift_data=res, model_metadata=self.model_metadata)


def _get_predictions_and_scores(model_metadata: ModelMetadata) -> Tuple[List[str], List[str]]:
    prediction_column_names: List[str] = []
    predicted_probabilities_column_names: List[str] = []

    # add continuous predictions or predicted probabilities from metadata to the selected features
    if isinstance(model_metadata, BinaryClassificationMetadata):
        if model_metadata.predicted_probability_column_name is None:
            raise MissingMetadataException(
                "missing value for 'predicted_probability_column_name'. "
                "Please update your model metadata accordingly."
            )
        prediction_column_names = []
        predicted_probabilities_column_names = [
            cast(BinaryClassificationMetadata, model_metadata).predicted_probability_column_name
        ]

    elif isinstance(model_metadata, MulticlassClassificationMetadata):
        if model_metadata.predicted_probabilities_column_names is None:
            raise MissingMetadataException(
                "missing value for 'predicted_probability_column_name'. "
                "Please update your model metadata accordingly."
            )
        md = cast(MulticlassClassificationMetadata, model_metadata)
        prediction_column_names = []
        predicted_probabilities_column_names = list(md.predicted_probabilities_column_names.values())

    elif isinstance(model_metadata, RegressionMetadata):
        if model_metadata.prediction_column_name is None:
            raise MissingMetadataException(
                "missing value for 'prediction_column_name'. " "Please update your model metadata accordingly."
            )
        prediction_column_names = [model_metadata.prediction_column_name]
        predicted_probabilities_column_names = []

    return prediction_column_names, predicted_probabilities_column_names
