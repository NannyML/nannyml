.. _deep_dive_metadata_extraction:

===================
Metadata extraction
===================

Naming conventions
------------------

NannyML uses some simple naming conventions to recognize metadata columns.
By using the right column names in your dataset, NannyML can extract all required metadata automatically.

To find out how to perform metadata extraction, check out :ref:`our tutorial on extracting metadata<import-data>`. 
Below you will find the naming conventions we use to identify each column.


Common metadata columns
^^^^^^^^^^^^^^^^^^^^^^^

Timestamp
""""""""""

.. code-block:: python

    def _guess_timestamps(data: pd.DataFrame) -> List[str]:
        def _guess_if_timestamp(col: pd.Series) -> bool:
            return col.name in ['date', 'timestamp', 'ts', 'date', 'time']

        return [col for col in data.columns if _guess_if_timestamp(data[col])]

Target
"""""""

.. code-block:: python

    def _guess_targets(data: pd.DataFrame) -> List[str]:
        def _guess_if_ground_truth(col: pd.Series) -> bool:
            return col.name in ['target', 'ground_truth', 'actual', 'actuals', 'y_true']

        return [col for col in data.columns if _guess_if_ground_truth(data[col])]

Partition
""""""""""

.. code-block:: python

    def _guess_partitions(data: pd.DataFrame) -> List[str]:
        def _guess_if_partition(col: pd.Series) -> bool:
            return 'partition' in col.name

        return [col for col in data.columns if _guess_if_partition(data[col])]

Binary classification columns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Predicted score/probability
"""""""""""""""""""""""""""""""""""""

.. code-block:: python

    def _guess_predictions(data: pd.DataFrame) -> List[str]:
        def _guess_if_prediction(col: pd.Series) -> bool:
            return col.name in ['p', 'pred', 'prediction', 'out', 'output', 'y_pred']

        return [col for col in data.columns if _guess_if_prediction(data[col])]


Predicted label
""""""""""""""""""""""""""

.. code-block:: python

    def _guess_predicted_probabilities(data: pd.DataFrame) -> List[str]:
        def _guess_if_prediction(col: pd.Series) -> bool:
            return col.name in ['y_pred_proba']

        return [col for col in data.columns if _guess_if_prediction(data[col])]

Multiclass classification columns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Predicted class scores/probabilities
""""""""""""""""""""""""""""""""""""""""

.. code-block:: python

    def _guess_predicted_probabilities(data: pd.DataFrame) -> List[str]:
        def _guess_if_prediction(col: pd.Series) -> bool:
            return str(col.name).startswith(PREDICTED_PROBABILITIES_PATTERN)

        return [col for col in data.columns if _guess_if_prediction(data[col])]


Predicted class label
""""""""""""""""""""""""""

.. code-block:: python

    def _guess_predicted_probabilities(data: pd.DataFrame) -> List[str]:
        def _guess_if_prediction(col: pd.Series) -> bool:
            return col.name in ['y_pred_proba']

        return [col for col in data.columns if _guess_if_prediction(data[col])]

Feature type detection
-----------------------

.. code-block:: python

    # When there are is not enough data to deduce anything
    if row_count < INFERENCE_NUM_ROWS_THRESHOLD:
        return FeatureType.UNKNOWN

    # If the values are floats, the feature is likely continuous
    if data_type == 'float64':
        return FeatureType.CONTINUOUS

    # If a high number of all values are unique, the feature is likely continuous
    if unique_fraction >= INFERENCE_HIGH_CARDINALITY_THRESHOLD:
        return FeatureType.CONTINUOUS

    # If a low enough number of the values are unique, the feature is likely categorical
    elif INFERENCE_LOW_CARDINALITY_THRESHOLD <= unique_fraction <= INFERENCE_MEDIUM_CARDINALITY_THRESHOLD:
        return FeatureType.CATEGORICAL

    # In any other case any there is not enough certainty
    else:
        return FeatureType.UNKNOWN
