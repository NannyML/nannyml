.. _import-data:

==========
Setting up
==========

.. note::
    A lot of new terminology will be introduced here. A full terminology overview is available in our
    :ref:`glossary<glossary>`.

This guide illustrates how to set up NannyML for use on a given model. Some data for a fictional model is available.
The **work_from_home** model is a binary classifier trying to predict whether someone will be working from home on
a given day or not. You can see below how to import it and explore it.

.. code-block:: python

    >>> import nannyml as nml
    >>> reference, analysis, analysis_gt = nml.load_synthetic_sample()
    >>> reference
           distance_from_office salary_range  gas_price_per_litre  public_transportation_cost  ...  work_home_actual            timestamp  y_pred_proba  partition
    0                  5.962247  40K - 60K €             2.119485                    8.568058  ...                 1  2014-05-09 22:27:20          0.99  reference
    1                  0.535872  40K - 60K €             2.357199                    5.425382  ...                 0  2014-05-09 22:59:32          0.07  reference
    2                  1.969519  40K - 60K €             2.366849                    8.247158  ...                 1  2014-05-09 23:48:25          1.00  reference
    3                  2.530410  20K - 20K €             2.318722                    7.944251  ...                 1  2014-05-10 01:12:09          0.98  reference
    4                  2.253635       60K+ €             2.221265                    8.884478  ...                 1  2014-05-10 02:21:34          0.99  reference
    ...                     ...          ...                  ...                         ...  ...               ...                  ...           ...        ...
    49995              2.356053  20K - 20K €             2.344472                    8.763312  ...                 1  2017-08-31 00:32:42          0.99  reference
    49996              4.897964    0 - 20K €             1.601283                    8.795181  ...                 0  2017-08-31 01:57:54          0.03  reference
    49997              0.869910  40K - 60K €             2.262292                    8.360564  ...                 1  2017-08-31 02:34:28          0.98  reference
    49998             10.336794    0 - 20K €             1.516446                    8.733694  ...                 0  2017-08-31 03:10:27          0.00  reference
    49999              2.815616  20K - 20K €             2.244124                    7.473265  ...                 1  2017-08-31 03:10:29          1.00  reference

On data, metadata and observations
==================================

NannyML offers tools to help monitor **models** in production.
To do so it needs the understand what the :term:`Model inputs`,
:term:`Model predictions` and :term:`Target` are for the monitored model.
NannyML will leverage this information to know which columns to use and how to interpret the data in them for
drift calculation and performance estimation.

For this purpose NannyML also requires some additional data, columns that are not features of the monitored model,
such as a timestamp or partition data.

The information describing the features, prediction, target and the columns
where to find required data is called :term:`Model Metadata`.

..
    TODO: insert illustration showing model invocation and assigning names to everything

When NannyML is used on a dataset it treats each row of that set as an *observation*: a combination of metadata,
feature inputs, the model output and target.

..
    TODO: insert illustration that shows all data in tabular form with annotations


Data requirements
=================

The data provided to NannyML should contain the following columns:

Identifier
----------

A unique, identifying value for each observation. See :term:`Identifier`.

If the data does not contain any real identifier column an artificial one can always be created. Row numbers or
timestamps are good candidates.

Timestamp
---------

Name of the column containing the timestamp at which the observation occurred, i.e. when the model was invoked
using the given inputs and yielding the resulting prediction. See :term:`Timestamp`.

.. note::
            **Format**
                Any format supported by Pandas, most likely:

                - *ISO 8601*, e.g. ``2021-10-13T08:47:23Z``
                - *Unix-epoch* in units of seconds, e.g. ``1513393355``

Predicted probability
---------------------

The score or probability that is emitted by the model, most likely a float.

.. warning::
    Either this or the *prediction* should be set for the metadata to be complete.


Prediction
----------

The predicted label, retrieved by interpreting (thresholding) the prediction scores or probabilities.

.. warning::
    Either this property or the *predicted_probability* should be set for the metadata to be complete.

.. warning::
    In case of binary prediction the *prediction values* should be either ``0`` or ``1`` for all functionality to work
    as intended. NannyML will interpret ``1`` as the *positive label*.

Target
------

The actual outcome of the event the machine learning model is trying to predict. See :term:`Target`.

.. note::
    **Target** values are only required in the reference data. Performance will be *calculated* using them.
    In the *analysis data* where they are not required, performance can be *estimated*.

Partition
---------

The partition each observation belongs to, an indicator for NannyML on whether to use this observation as
*reference* data or *analysis* data. The *reference* data contains observations for which target values
are available, hence the model performance can be *calculated* for this set.
The occurrence of drift - or the lack hereof - is known and validated.
The *analysis* data does not contain any target values, hence performance can only be *estimated*.

----

This means that for the example **work_from_home** case:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Metadata property
     - Value
   * - Features
     - ``nationality``

       ``dist_to_office``

       ``day_of_the_week``

       ``work_home_yest``

       ``transport_mode``

       ``industry``
   * - Predicted probability
     - ``y_pred_proba``
   * - Prediction
     - ``np.NaN``
   * - Ground truth.
     - ``work_home_actual``
   * - Identifier
     - ``identifier``
   * - Timestamp
     - ``timestamp``
   * - Partition
     - ``partition``

Providing metadata
===================

NannyML uses the :class:`nannyml.metadata.ModelMetadata` and :class:`nannyml.metadata.Feature` classes
to deal with metadata. Whilst it is possible to construct the model metadata fully manual using these classes,
this approach does not scale well for more complex models with many features.

Extracting metadata
-------------------

NannyML provides the :func:`nannyml.metadata.extract_metadata` function to automatically extract the required metadata
from a given ``DataFrame``. It does so by following some simple naming conventions and heuristics to column names
and data. It returns a prepopulated instance of the :class:`ModelMetadata<nannyml.metadata.ModelMetadata>` class.

.. code-block:: python

    >>> metadata = nml.extract_metadata(data=reference)
    >>> metadata
    Metadata({'identifier_column_name': 'identifier', 'timestamp_column_name': 'timestamp', 'partition_column_name': 'partition', 'target_column_name': None, 'prediction_column_name': 'y_pred_proba', 'features': "[Feature({'label': 'distance_from_office', 'column_name': 'distance_from_office', 'type': 'continuous', 'description': 'extracted feature: distance_from_office'}), Feature({'label': 'salary_range', 'column_name': 'salary_range', 'type': 'categorical', 'description': 'extracted feature: salary_range'}), Feature({'label': 'gas_price_per_litre', 'column_name': 'gas_price_per_litre', 'type': 'continuous', 'description': 'extracted feature: gas_price_per_litre'}), Feature({'label': 'public_transportation_cost', 'column_name': 'public_transportation_cost', 'type': 'continuous', 'description': 'extracted feature: public_transportation_cost'}), Feature({'label': 'wfh_prev_workday', 'column_name': 'wfh_prev_workday', 'type': 'categorical', 'description': 'extracted feature: wfh_prev_workday'}), Feature({'label': 'workday', 'column_name': 'workday', 'type': 'categorical', 'description': 'extracted feature: workday'}), Feature({'label': 'tenure', 'column_name': 'tenure', 'type': 'continuous', 'description': 'extracted feature: tenure'}), Feature({'label': 'work_home_actual', 'column_name': 'work_home_actual', 'type': 'categorical', 'description': 'extracted feature: work_home_actual'})]"})

The metadata can then be printed using the :meth:`nannyml.metadata.ModelMetadata.print` method or returned as a
``dictionary`` or a ``DataFrame``.

.. code-block:: python

    >>> metadata.print()
    Metadata for model ~ UNKNOWN ~
    # Warning - unable to identify all essential data
    # Please identify column names for all '~ UNKNOWN ~' values
    Model problem             binary_classification
    Identifier column         identifier
    Timestamp column          timestamp
    Partition column          partition
    Prediction column         y_pred_proba
    Prediction column         ~ UNKNOWN ~
    Target column             ~ UNKNOWN ~

    Features

    Name                        Column                      Type            Description
    distance_from_office        distance_from_office        continuous      extracted feature: distance_from_office
    salary_range                salary_range                categorical     extracted feature: salary_range
    gas_price_per_litre         gas_price_per_litre         continuous      extracted feature: gas_price_per_litre
    public_transportation_cost  public_transportation_cost  continuous      extracted feature: public_transportation_cost
    wfh_prev_workday            wfh_prev_workday            categorical     extracted feature: wfh_prev_workday
    workday                     workday                     categorical     extracted feature: workday
    tenure                      tenure                      continuous      extracted feature: tenure
    work_home_actual            work_home_actual            categorical     extracted feature: work_home_actual

    >>> metadata.to_dict()
    {'identifier_column_name': 'identifier',
     'timestamp_column_name': 'timestamp',
     'partition_column_name': 'partition',
     'target_column_name': None,
     'prediction_column_name': 'y_pred_proba',
     'features': "[Feature({'label': 'distance_from_office', 'column_name': 'distance_from_office', 'type': 'continuous', 'description': 'extracted feature: distance_from_office'}), Feature({'label': 'salary_range', 'column_name': 'salary_range', 'type': 'categorical', 'description': 'extracted feature: salary_range'}), Feature({'label': 'gas_price_per_litre', 'column_name': 'gas_price_per_litre', 'type': 'continuous', 'description': 'extracted feature: gas_price_per_litre'}), Feature({'label': 'public_transportation_cost', 'column_name': 'public_transportation_cost', 'type': 'continuous', 'description': 'extracted feature: public_transportation_cost'}), Feature({'label': 'wfh_prev_workday', 'column_name': 'wfh_prev_workday', 'type': 'categorical', 'description': 'extracted feature: wfh_prev_workday'}), Feature({'label': 'workday', 'column_name': 'workday', 'type': 'categorical', 'description': 'extracted feature: workday'}), Feature({'label': 'tenure', 'column_name': 'tenure', 'type': 'continuous', 'description': 'extracted feature: tenure'}), Feature({'label': 'work_home_actual', 'column_name': 'work_home_actual', 'type': 'categorical', 'description': 'extracted feature: work_home_actual'})]"}

    >>> metadata.to_df()
                                 label  ...                                    description
    0       identifier_column_name  ...                                     identifier
    1        timestamp_column_name  ...                                      timestamp
    2        partition_column_name  ...                                      partition
    3           target_column_name  ...                                         target
    4       prediction_column_name  ...                   prediction score/probability
    5         distance_from_office  ...        extracted feature: distance_from_office
    6                 salary_range  ...                extracted feature: salary_range
    7          gas_price_per_litre  ...         extracted feature: gas_price_per_litre
    8   public_transportation_cost  ...  extracted feature: public_transportation_cost
    9             wfh_prev_workday  ...            extracted feature: wfh_prev_workday
    10                     workday  ...                     extracted feature: workday
    11                      tenure  ...                      extracted feature: tenure
    12            work_home_actual  ...            extracted feature: work_home_actual

.. warning::
    Because the extraction is based on simple rules the results are never guaranteed to be completely correct.
    It is strongly advised to review the results of :func:`extract_metadata<nannyml.metadata.extract_metadata>`
    and update the values where needed.

Heuristics
----------

NannyML uses some simple heuristics to detect metadata, often by naming convention. By using the right column names,
NannyML can extract all required metadata automatically.

These metadata properties follow simple naming conventions for discovery:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Metadata property
     - Naming convention
   * - ``identifier_column_name``
     - ``column_name in ['id', 'ident', 'identity', 'identifier', 'uid', 'uuid']``
   * - ``timestamp_column_name``
     - ``column_name in ['date', 'timestamp', 'ts', 'date', 'time']``
   * - ``predicted_probability_column_name``
     - ``column_name in ['y_pred_proba']``
   * - ``prediction_column_name``
     - ``column_name in ['p', 'pred', 'prediction', 'out', 'output', 'y_pred']``
   * - ``target_column_name``
     - ``column_name in ['target', 'ground_truth', 'actual', 'actuals']``
   * - ``partition_column_name``
     - ``column_name in ['partition']``

Any column not flagged as one of the above is considered to be a feature. To assign the appropriate
:class:`feature type<nannyml.metadata.FeatureType>` NannyML will evaluate the feature values and apply
the following heuristic:

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

NannyML will raise exceptions when trying to run calculations with incomplete metadata, i.e. when not all properties
were provided. NannyML includes a quick way to check if the metadata is fully completed.

The :meth:`nannyml.metadata.Metadata.is_complete` method will check a :class:`ModelMetadata<nannyml.metadata.ModelMetadata>`
instance and return a tuple. The first element - a boolean - is the answer to the *is complete* question.
The second element - an array - represents the properties that are still missing.

We can see in our example that we are currently missing the ``target_column_name``.

.. code-block:: python

    >>> metadata.is_complete()
    (False, ['target_column_name'])

Updating metadata
-----------------

The metadata can be completed by providing the missing value.

.. code-block:: python

    >>> metadata.target_column_name = 'work_home_actual'
    >>> metadata.is_complete()
    (True, [])  # yay, our metadata is all good to go!

It looks like the metadata is now complete and ready to use in
:ref:`drift calculation<data-drift>` or :ref:`performance estimation<performance-estimation>`.
