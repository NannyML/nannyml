.. _import-data:

==================
Providing metadata
==================

Why is providing metadata required?
=======================================

NannyML can process any data used in supported models. It requires model metadata to
assign the correct role to each column of the data set. You can provide a
:class:`~nannyml.metadata.base.ModelMetadata` object that allows NannyML to make sense of your data.
It allows you to specify what the :term:`model inputs<Model inputs>`, :term:`model predictions<Model predictions>`
and :term:`targets <Target>` are for your monitored model.

This guide will illustrate how to use NannyML to help your create this
:class:`~nannyml.metadata.base.ModelMetadata` object.


Metadata for binary classification
======================================

We'll use the same sample dataset for this guide as we use in our :ref:`quick start<quick-start>`.
The dataset describes a machine learning model that tries to predict whether an employee will work from
home on the next day, and is included in the library.
You can read more about it on the :ref:`dataset introduction page<dataset-synthetic-binary>`.


Just the code
-------------

.. code-block:: python

    >>> import nannyml as nml
    >>> reference, analysis, analysis_targets = nml.load_synthetic_binary_classification_dataset()
    >>> reference.columns
    Index(['distance_from_office', 'salary_range', 'gas_price_per_litre',
       'public_transportation_cost', 'wfh_prev_workday', 'workday', 'tenure',
       'identifier', 'work_home_actual', 'timestamp', 'y_pred_proba',
       'partition', 'y_pred'],
      dtype='object')
    >>> reference.head()

    >>> metadata = nml.extract_metadata(data=reference, model_type='classification_binary', exclude_columns=['identifier'])
    >>> metadata.is_complete()
    (False, ['target_column_name'])
    >>> metadata.target_column_name = 'work_home_actual'
    >>> metadata.is_complete()
    (True, [])
    >>> metadata.to_df()


Walkthrough
-----------

First we load the demo data. This returns three different ``DataFrames``. The first two correspond to
the different :term:`data periods<Data Period>`, containing the data of the reference and analysis periods.
The third ``DataFrame`` contains the target values for the analysis period. It can be joined with this period by
using the shared ``identifier`` column.

.. code-block:: python

    >>> import nannyml as nml
    >>> reference, analysis, analysis_targets = nml.load_synthetic_binary_classification_dataset()

-----

Next we inspect the data inside the reference period.

The ``y_pred`` and ``y_pred_proba`` columns are housing the predicted labels and prediction scores or
probabilities, i.e. the model outputs.

The ``work_home_actual`` column contains the target values (remember, we're looking at the *reference*
period here, for which target values are available).

The ``partition`` column contains the name of the :term:`data period<Data Period>` the observation belongs to, in this
case all of them belong to the *reference* period.

The ``timestamp`` column contains the timestamp at which the model did this particular prediction.

The ``identifier`` column is used to uniquely identify each row. It is not a feature as it does not serve as an input
for the model.

The rest of the columns are the model inputs containing either continuous or categorical feature values.

.. code-block:: python

    >>> Index(['distance_from_office', 'salary_range', 'gas_price_per_litre',
       'public_transportation_cost', 'wfh_prev_workday', 'workday', 'tenure',
       'identifier', 'work_home_actual', 'timestamp', 'y_pred_proba',
       'partition', 'y_pred'],
      dtype='object')

-----

We can now leverage the :func:`nannyml.metadata.extraction.extract_metadata` function to create a
:class:`~nannyml.metadata.base.ModelMetadata` object from the reference data.

The ``data`` argument is used to pass the data sample for the extraction.

The ``model_type`` argument allows us to specify the type of the model that is monitored -
either ``classification_binary`` or ``classification_multiclass``.

The exact algorithm does not matter, as NannyML doesn't use the model when analysing data.
This argument allows the :func:`nannyml.metadata.extraction.extract_metadata`
function to look for specific patterns in the columns, based on the type of model specified.

The ``exclude_columns`` argument is used to identify columns that are not relevant to the model.
In this example case the ``identifier`` column is only used as a helper to perform the join
between the analysis period data and its target values. By excluding it we can ensure it is not picked up as a
model feature by NannyML.

.. code-block:: python

    >>> metadata = nml.extract_metadata(data=reference, model_type='classification_binary', exclude_columns=['identifier'])


-----

The :func:`nannyml.metadata.extraction.extract_metadata` function uses some simple heuristics to yield its results.
You can read more on the inner workings of this function in :ref:`metadata extraction <deep_dive_metadata_extraction>`
This means that in some cases it will not succeed in extracting all required information.

The :func:`nannyml.metadata.base.is_complete` function checks if all required metadata properties have been provided.
It is normally used internally to validate user inputs. The function returns a ``bool`` indicating if metadata is
complete. The second return argument is an array containing the name of any missing properties.
Running this step is not necessary but can be done to double-check everything is in order before doing anything else.

We can see that the extraction was not able to find the ``target_column_name``, i.e. the column containing the target
values (``work_home_actual``) in our case.

.. code-block:: python

    >>> metadata.is_complete()
    (False, ['target_column_name'])


-----

We can fix this missing metadata by modifying the :class:`~nannyml.metadata.base.ModelMetadata` object returned by the
:func:`nannyml.metadata.extraction.extract_metadata` function to set its ``target_column_name`` property.

.. code-block:: python

    >>> metadata.target_column_name = 'work_home_actual'

.. note::
    All :class:`~nannyml.metadata.binary_classification.BinaryClassificationMetadata` properties can be updated
    when they are missing or incorrect.

    These are:
        - ``target_column_name``
        - ``partition_column_name``
        - ``timestamp_column_name``
        - ``prediction_column_name``
        - ``predicted_probability_column_name``

-----

We see the metadata is now considered complete. We can represent the :class:`~nannyml.metadata.base.ModelMetadata`
object as a ``DataFrame`` for easy inspection.

.. code-block:: python

    >>> metadata.is_complete()
    (True, [])
    >>> metadata.to_df()

+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+
|    | label                             | column_name                | type        | description                                   |
+====+===================================+============================+=============+===============================================+
|  0 | timestamp_column_name             | timestamp                  | continuous  | timestamp                                     |
+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+
|  1 | partition_column_name             | partition                  | categorical | partition                                     |
+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+
|  2 | target_column_name                | work_home_actual           | categorical | target                                        |
+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+
|  3 | distance_from_office              | distance_from_office       | continuous  | extracted feature: distance_from_office       |
+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+
|  4 | salary_range                      | salary_range               | categorical | extracted feature: salary_range               |
+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+
|  5 | gas_price_per_litre               | gas_price_per_litre        | continuous  | extracted feature: gas_price_per_litre        |
+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+
|  6 | public_transportation_cost        | public_transportation_cost | continuous  | extracted feature: public_transportation_cost |
+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+
|  7 | wfh_prev_workday                  | wfh_prev_workday           | categorical | extracted feature: wfh_prev_workday           |
+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+
|  8 | workday                           | workday                    | categorical | extracted feature: workday                    |
+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+
|  9 | tenure                            | tenure                     | continuous  | extracted feature: tenure                     |
+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+
| 10 | prediction_column_name            | y_pred                     | continuous  | predicted label                               |
+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+
| 11 | predicted_probability_column_name | y_pred_proba               | continuous  | predicted score/probability                   |
+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+


Metadata for multiclass classification
=======================================

We'll use a sample dataset for this guide. The dataset describes a machine learning model that tries to predict
the most appropriate product for new customers applying for a credit card.
You can read more about it on the :ref:`dataset introduction page<dataset-synthetic-multiclass>`.

Just the code
-------------

.. code-block:: python

    >>> import nannyml as nml
    >>> reference, analysis, analysis_targets = nml.load_synthetic_multiclass_classification_dataset()
    >>> reference.columns
    Index(['acq_channel', 'app_behavioral_score', 'requested_credit_limit',
       'app_channel', 'credit_bureau_score', 'stated_income', 'is_customer',
       'partition', 'identifier', 'timestamp', 'y_pred_proba_prepaid_card',
       'y_pred_proba_highstreet_card', 'y_pred_proba_upmarket_card', 'y_pred',
       'y_true'],
      dtype='object')
    >>> reference.head()

    >>> metadata = nml.extract_metadata(data=reference, model_type='classification_multiclass', exclude_columns=['identifier'])
    >>> metadata.is_complete()
    (True, [])
    >>> metadata.to_df()

.. code-block:: python

    >>> metadata.predicted_probabilities_column_names
    {'prepaid_card': 'y_pred_proba_prepaid_card',
     'highstreet_card': 'y_pred_proba_highstreet_card',
     'upmarket_card': 'y_pred_proba_upmarket_card'}

Walkthrough
-----------

The first line loads the demo data. This returns three different ``DataFrames``. The first two correspond to
the different :term:`data periods<Data Period>`, containing the data of the reference and analysis periods.
The third ``DataFrame`` contains the target values for the analysis period. It can be joined with this period by
using the shared ``identifier`` column.

.. code-block:: python

    >>> import nannyml as nml
    >>> reference, analysis, analysis_targets = nml.load_synthetic_multiclass_classification_dataset()

-----

Next we inspect the data inside the reference period.

The ``y_pred`` column contains the labels predicted by the model.

The ``y_pred_proba_prepaid_card``, ``y_pred_proba_highstreet_card`` and ``y_pred_proba_upmarket_card``
contain the predicted class probabilities for the three classes labeled ``prepaid_card``, ``highstreet_card``
and ``upmarket_card``.

The ``y_true`` column contains the target values (remember, we're looking at the *reference*
period here, for which target values are available).

The ``partition`` column contains the name of the :term:`data period<Data Period>` the observation belongs to, in this
case all of them belong to the *reference* period.

The ``timestamp`` column contains the timestamp at which the model did this particular prediction.

The ``identifier`` column is used to uniquely identify each row. It is not a feature as it does not serve as an input
for the model.

The rest of the columns are the model inputs containing either continuous or categorical feature values.

.. code-block:: python

    >>> Index(['acq_channel', 'app_behavioral_score', 'requested_credit_limit',
       'app_channel', 'credit_bureau_score', 'stated_income', 'is_customer',
       'partition', 'identifier', 'timestamp', 'y_pred_proba_prepaid_card',
       'y_pred_proba_highstreet_card', 'y_pred_proba_upmarket_card', 'y_pred',
       'y_true'],
      dtype='object')



-----

We can now leverage the :func:`nannyml.metadata.extraction.extract_metadata` function to create a
:class:`~nannyml.metadata.base.ModelMetadata` object from the reference data.

The ``data`` argument is used to pass the data sample for the extraction.

The ``model_type``The model_type argument allows us to specify the type of the model that is monitored -
either ``classification_binary`` or ``classification_multiclass``.
The exact algorithm does not matter, as NannyML doesn't use the model when analysing data.
This argument allows the :func:`nannyml.metadata.extraction.extract_metadata`
function to look for specific patterns in the columns.

The ``exclude_columns`` argument is used to pass along the names of columns that are not relevant to the model.
In this example case the ``identifier`` column is such a column: it is only used as a helper to perform the join
between the *analysis* period data and its target values. By excluding it we can ensure it is not picked up as a
model feature by NannyML.

.. code-block:: python

    >>> metadata = nml.extract_metadata(data=reference, model_type='classification_multiclass', exclude_columns=['identifier'])



-----

The :func:`nannyml.metadata.extraction.extract_metadata` function uses some simple heuristics to yield its results.
You can read more on the inner workings of this function in :ref:`metadata extraction <deep_dive_metadata_extraction>`
This means that in some cases it will not succeed in extracting all required information.

The :func:`nannyml.metadata.base.is_complete` function checks if all required metadata properties have been provided.
The function returns a ``bool`` indicating if metadata is complete. The second return argument is an array 
containing the name of any missing properties. Running this step is not necessary but can be done to 
double-check everything is in order.

.. code-block:: python

    >>> metadata.is_complete()
    (True, [])

We can see that the extraction was able to find all required properties. The metadata is considered to be complete.

.. note::
    All :class:`~nannyml.metadata.multiclass_classification.MulticlassClassificationMetadata` properties can be updated
    when they are missing or incorrect.

    These are:
        - ``target_column_name``
        - ``partition_column_name``
        - ``timestamp_column_name``
        - ``prediction_column_name``
        - ``predicted_probabilities_column_names``

-----

We can represent the :class:`~nannyml.metadata.base.ModelMetadata` object as a ``DataFrame`` for easy inspection.

.. code-block:: python

    >>> metadata.is_complete()
    (True, [])
    >>> metadata.to_df()

+----+---------------------------------------------------+------------------------------+-------------+---------------------------------------------------------+
|    | label                                             | column_name                  | type        | description                                             |
+====+===================================================+==============================+=============+=========================================================+
|  0 | timestamp_column_name                             | timestamp                    | continuous  | timestamp                                               |
+----+---------------------------------------------------+------------------------------+-------------+---------------------------------------------------------+
|  1 | partition_column_name                             | partition                    | categorical | partition                                               |
+----+---------------------------------------------------+------------------------------+-------------+---------------------------------------------------------+
|  2 | target_column_name                                | y_true                       | categorical | target                                                  |
+----+---------------------------------------------------+------------------------------+-------------+---------------------------------------------------------+
|  3 | acq_channel                                       | acq_channel                  | categorical | extracted feature: acq_channel                          |
+----+---------------------------------------------------+------------------------------+-------------+---------------------------------------------------------+
|  4 | app_behavioral_score                              | app_behavioral_score         | continuous  | extracted feature: app_behavioral_score                 |
+----+---------------------------------------------------+------------------------------+-------------+---------------------------------------------------------+
|  5 | requested_credit_limit                            | requested_credit_limit       | categorical | extracted feature: requested_credit_limit               |
+----+---------------------------------------------------+------------------------------+-------------+---------------------------------------------------------+
|  6 | app_channel                                       | app_channel                  | categorical | extracted feature: app_channel                          |
+----+---------------------------------------------------+------------------------------+-------------+---------------------------------------------------------+
|  7 | credit_bureau_score                               | credit_bureau_score          | continuous  | extracted feature: credit_bureau_score                  |
+----+---------------------------------------------------+------------------------------+-------------+---------------------------------------------------------+
|  8 | stated_income                                     | stated_income                | categorical | extracted feature: stated_income                        |
+----+---------------------------------------------------+------------------------------+-------------+---------------------------------------------------------+
|  9 | is_customer                                       | is_customer                  | categorical | extracted feature: is_customer                          |
+----+---------------------------------------------------+------------------------------+-------------+---------------------------------------------------------+
| 10 | prediction_column_name                            | y_pred                       | continuous  | predicted label                                         |
+----+---------------------------------------------------+------------------------------+-------------+---------------------------------------------------------+
| 11 | predicted_probability_column_name_prepaid_card    | y_pred_proba_prepaid_card    | continuous  | predicted score/probability for class 'prepaid_card'    |
+----+---------------------------------------------------+------------------------------+-------------+---------------------------------------------------------+
| 12 | predicted_probability_column_name_highstreet_card | y_pred_proba_highstreet_card | continuous  | predicted score/probability for class 'highstreet_card' |
+----+---------------------------------------------------+------------------------------+-------------+---------------------------------------------------------+
| 13 | predicted_probability_column_name_upmarket_card   | y_pred_proba_upmarket_card   | continuous  | predicted score/probability for class 'upmarket_card'   |
+----+---------------------------------------------------+------------------------------+-------------+---------------------------------------------------------+

-----

We can now inspect the :class:`~nannyml.metadata.multiclass_classification.MulticlassClassificationMetadata` object
and find the mapping of class labels to a predicted probability column for that class, stored as a Python ``dict``.

.. code-block:: python

    >>> metadata.predicted_probabilities_column_names
    {'prepaid_card': 'y_pred_proba_prepaid_card',
     'highstreet_card': 'y_pred_proba_highstreet_card',
     'upmarket_card': 'y_pred_proba_upmarket_card'}

Metadata for regression
======================================

.. warning::

    This is an early release that is using a modified version of the :ref:`binary classification dataset<dataset-synthetic-binary>`. A proper
    sample data set for regression, suitable for running performance calculation and estimation will released later.

    This sample is sufficient to illustrate data preparation for regression cases to be used in drift detection.


We'll use the same sample dataset for this guide as we use in our :ref:`quick start<quick-start>`.
The dataset describes a machine learning model that tries to predict whether an employee will work from
home on the next day, and is included in the library.
You can read more about it on the :ref:`dataset introduction page<dataset-synthetic-binary>`.


Just the code
-------------

.. code-block:: python

    >>> import nannyml as nml
    >>> reference, analysis, analysis_targets = nml.load_synthetic_binary_classification_dataset()
    >>> reference = ref_df.drop(columns=['y_pred_proba'])
    >>> analysis = ana_df.drop(columns=['y_pred_proba'])
    >>> reference['y_pred'] = np.random.randn(len(ref_df))
    >>> analysis['y_pred'] = np.random.randn(len(ref_df))
    >>> reference.columns
    Index(['distance_from_office', 'salary_range', 'gas_price_per_litre',
       'public_transportation_cost', 'wfh_prev_workday', 'workday', 'tenure',
       'identifier', 'work_home_actual', 'timestamp', 'partition', 'y_pred'],
      dtype='object')
    >>> reference.head()

    >>> metadata = nml.extract_metadata(data=reference, model_type='regression', exclude_columns=['identifier'])
    >>> metadata.is_complete()
    (False, ['target_column_name'])
    >>> metadata.target_column_name = 'work_home_actual'
    >>> metadata.is_complete()
    (True, [])
    >>> metadata.to_df()


Walkthrough
-----------

The first line loads the demo data. This returns three different ``DataFrames``. The first two correspond to
the different :term:`data periods<Data Period>`, containing the data of the reference and analysis periods.
The third ``DataFrame`` contains the target values for the analysis period. It can be joined with this period by
using the shared ``identifier`` column.

.. note::

    We are manually altering the existing :ref:`binary classification dataset<dataset-synthetic-binary>` 
    to make it appear to be data from a regression model.
    First we drop the predicted probabilities from both the reference and analysis periods.
    Then we set the predictions to be random `float` values.
    If you have a regression model dataset already, this is obviously unnecessary.

.. code-block:: python

    >>> import nannyml as nml
    >>> reference, analysis, analysis_targets = nml.load_synthetic_binary_classification_dataset()
    >>> reference = ref_df.drop(columns=['y_pred_proba'])
    >>> analysis = ana_df.drop(columns=['y_pred_proba'])
    >>> reference['y_pred'] = np.random.randn(len(ref_df))
    >>> analysis['y_pred'] = np.random.randn(len(ref_df))

-----

Next we inspect the data inside the reference period.

The ``y_pred`` column is housing the predictions, i.e. the model outputs.

The ``work_home_actual`` column contains the target values (remember, we're looking at the *reference*
period here, for which target values are available).

The ``partition`` column contains the name of the :term:`data period<Data Period>` the observation belongs to, in this
case all of them belong to the *reference* period.

The ``timestamp`` column contains the timestamp at which the model did this particular prediction.

The ``identifier`` column is used to uniquely identify each row. It is not a feature as it does not serve as an input
for the model.

The rest of the columns are the model inputs containing either continuous or categorical feature values.

.. code-block:: python

    >>> Index(['distance_from_office', 'salary_range', 'gas_price_per_litre',
       'public_transportation_cost', 'wfh_prev_workday', 'workday', 'tenure',
       'identifier', 'work_home_actual', 'timestamp',
       'partition', 'y_pred'],
      dtype='object')


-----

We can now leverage the :func:`nannyml.metadata.extraction.extract_metadata` function to create a
:class:`~nannyml.metadata.base.ModelMetadata` object from the reference data.

The ``data`` argument is used to pass the data sample for the extraction.

The ``model_type``The model_type argument allows us to specify the type of the model that is monitored -
either ``classification_binary`` or ``classification_multiclass``.
The exact algorithm does not matter, as NannyML doesn't use the model when analysing data.
This argument allows the :func:`nannyml.metadata.extraction.extract_metadata`
function to look for specific patterns in the columns.

The ``exclude_columns`` argument is used to pass along the names of columns that are not relevant to the model.
In this example case the ``identifier`` column is such a column: it is only used as a helper to perform the join
between the *analysis* period data and its target values. By excluding it we can ensure it is not picked up as a
model feature by NannyML.

.. code-block:: python

    >>> metadata = nml.extract_metadata(data=reference, model_type='classification_binary', exclude_columns=['identifier'])


-----

The :func:`nannyml.metadata.extraction.extract_metadata` function uses some simple heuristics to yield its results.
You can read more on the inner workings of this function in :ref:`metadata extraction <deep_dive_metadata_extraction>`
This means that in some cases it will not succeed in extracting all required information.

The :func:`nannyml.metadata.base.is_complete` function checks if all required metadata properties have been provided.
It is normally used internally to validate user inputs. The function returns a ``bool`` indicating if metadata is
complete. The second return argument is an array containing the name of any missing properties.
Running this step is not necessary but can be done to double-check everything is in order in advance.

.. code-block:: python

    >>> metadata.is_complete()
    (False, ['target_column_name'])

We can see that the extraction was not able to find the ``target_column_name``, i.e. the column containing the target
values (``work_home_actual``) in our case.

-----


To fix this we can modify the set the ``target_column_name`` property of the :class:`~nannyml.metadata.base.ModelMetadata` 
object returned by the :func:`nannyml.metadata.extraction.extract_metadata` function.

.. code-block:: python

    >>> metadata.target_column_name = 'work_home_actual'

.. note::
    All :class:`~nannyml.metadata.binary_classification.BinaryClassificationMetadata` properties can be updated
    when they are missing or incorrect.

    These are:
        - ``target_column_name``
        - ``partition_column_name``
        - ``timestamp_column_name``
        - ``prediction_column_name``

-----

We see the metadata is now considered complete. We can represent the :class:`~nannyml.metadata.base.ModelMetadata`
object as a ``DataFrame`` for easy inspection.

.. code-block:: python

    >>> metadata.is_complete()
    (True, [])
    >>> metadata.to_df()

+----+----------------------------+----------------------------+-------------+-----------------------------------------------+
|    | label                      | column_name                | type        | description                                   |
+====+============================+============================+=============+===============================================+
|  0 | timestamp_column_name      | timestamp                  | continuous  | timestamp                                     |
+----+----------------------------+----------------------------+-------------+-----------------------------------------------+
|  1 | partition_column_name      | partition                  | categorical | partition                                     |
+----+----------------------------+----------------------------+-------------+-----------------------------------------------+
|  2 | target_column_name         | work_home_actual           | categorical | target                                        |
+----+----------------------------+----------------------------+-------------+-----------------------------------------------+
|  3 | distance_from_office       | distance_from_office       | continuous  | extracted feature: distance_from_office       |
+----+----------------------------+----------------------------+-------------+-----------------------------------------------+
|  4 | salary_range               | salary_range               | categorical | extracted feature: salary_range               |
+----+----------------------------+----------------------------+-------------+-----------------------------------------------+
|  5 | gas_price_per_litre        | gas_price_per_litre        | continuous  | extracted feature: gas_price_per_litre        |
+----+----------------------------+----------------------------+-------------+-----------------------------------------------+
|  6 | public_transportation_cost | public_transportation_cost | continuous  | extracted feature: public_transportation_cost |
+----+----------------------------+----------------------------+-------------+-----------------------------------------------+
|  7 | wfh_prev_workday           | wfh_prev_workday           | categorical | extracted feature: wfh_prev_workday           |
+----+----------------------------+----------------------------+-------------+-----------------------------------------------+
|  8 | workday                    | workday                    | categorical | extracted feature: workday                    |
+----+----------------------------+----------------------------+-------------+-----------------------------------------------+
|  9 | tenure                     | tenure                     | continuous  | extracted feature: tenure                     |
+----+----------------------------+----------------------------+-------------+-----------------------------------------------+
| 10 | prediction_column_name     | y_pred                     | continuous  | predicted value                               |
+----+----------------------------+----------------------------+-------------+-----------------------------------------------+


Insights
=======================

.. warning::
    Because the extraction is based on simple rules the results are never guaranteed to be completely correct.
    It is strongly advised to review the results of
    :func:`extract_metadata<nannyml.metadata.extraction.extract_metadata>` and update the values where needed.

    NannyML will raise an :class:`~nannyml.exceptions.MissingMetadataException` when trying to run any functionality
    using incomplete metadata.

.. note::
    We are aware that this boilerplate setup step creates some friction. We're actively working
    on reducing it.


What next
=======================

To find out more about the columns that should in your dataset, check out the
:ref:`data requirements<data_requirements>` documentation.

You can read the :ref:`how metadata extraction works<deep_dive_metadata_extraction>` to find out more about our
naming conventions and heuristics.

You can put your shiny new metadata to use in :ref:`drift calculation<data-drift>`, :ref:`performance calculation<performance-calculation>`
or :ref:`performance estimation<performance-estimation>`.