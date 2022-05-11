.. _import-data:

==================
Providing metadata
==================

Why is data preparation required?
---------------------------------

NannyML intends to work with any data set it is given. To do so it needs to understand it and
assign a correct role to each column of the data set.

This can be done using the concept of *model metadata*. As a user you can provide a
:class:`~nannyml.metadata.base.ModelMetadata` object that allows NannyML to make sense of your data.
It allows you to specify what the :term:`model inputs<Model inputs>`, :term:`model predictions<Model predictions>`
and :term:`targets <Target>` are for your monitored model.

This guide will illustrate how to use NannyML to help your create this mythical
:class:`~nannyml.metadata.base.ModelMetadata` object.

We'll use a sample data set for this guide.
This fictional **work_from_home** model is a binary classifier trying to predict whether someone will be
working from home on a given day or not.


Just the code
-------------

.. code-block:: python

    >>> import nannyml as nml
    >>> reference, analysis, analysis_gt = nml.load_synthetic_binary_classification_dataset()
    >>> reference.columns
    Index(['distance_from_office', 'salary_range', 'gas_price_per_litre',
       'public_transportation_cost', 'wfh_prev_workday', 'workday', 'tenure',
       'identifier', 'work_home_actual', 'timestamp', 'y_pred_proba',
       'partition', 'y_pred'],
      dtype='object')
    >>> reference.head()
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
    >>> metadata = nml.extract_metadata(data=reference, model_type=nml.ModelType.CLASSIFICATION_BINARY, exclude_columns=['identifier'])
    >>> metadata.is_complete()
    (False, ['target_column_name'])
    >>> metadata.target_column_name = 'work_home_actual'
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


Walkthrough
-----------

The first line loads the demo data. Remark that it returns three different ``DataFrames``. The first two correspond to
the different :term:`data periods<Data Period>`, containing the data of the *reference* and *analysis* periods.
The third ``DataFrame`` contains the *target* values for the *analysis* period. It can be joined with this period by
using the shared ``identifier`` column.

.. code-block:: python

    >>> import nannyml as nml
    >>> reference, analysis, analysis_gt = nml.load_synthetic_binary_classification_dataset()

-----

The next lines takes a quick peek at the data inside the *reference* period.

.. code-block:: python

    >>> Index(['distance_from_office', 'salary_range', 'gas_price_per_litre',
       'public_transportation_cost', 'wfh_prev_workday', 'workday', 'tenure',
       'identifier', 'work_home_actual', 'timestamp', 'y_pred_proba',
       'partition', 'y_pred'],
      dtype='object')

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

-----

We can now leverage the :func:`nannyml.metadata.extraction.extract_metadata` function to create a
:class:`~nannyml.metadata.base.ModelMetadata` object from the *reference* data.

.. code-block:: python

    >>> metadata = nml.extract_metadata(data=reference, model_type=nml.ModelType.CLASSIFICATION_BINARY, exclude_columns=['identifier'])

The ``data`` argument is used to pass the data sample for the extraction.

The ``model_type`` argument allows us to specify the type of the model that is monitored. This is not about the
underlying implementation (e.g. logistic regression or an SVM) but about the problem that it tries to solve,
binary classification in this case. This argument allows the :func:`nannyml.metadata.extraction.extract_metadata`
function to look for specific patterns in the columns. Think about how the columns containing prediction scores or
probabilities will be different between binary classification or multiclass classification.

The ``exclude_columns`` argument is used to pass along the names of columns that are not relevant to the model at all.
In this example case the ``identifier`` column is such a column: it is solely used as a helper to perform the join
between the *analysis* period data and its *target* values. By excluding it we can ensure it is not picked up as a
model feature by NannyML.

-----

The next line is not really required and used for educational purposes here. The
:func:`nannyml.metadata.base.is_complete` function checks if all required metadata properties have been provided.
It is normally used internally to validate user inputs. The function returns a ``bool`` indicating if metadata is
complete. The second return argument is an array containing the name of any missing properties.

.. code-block:: python

    >>> metadata.is_complete()
    (False, ['target_column_name'])

We can see that the extraction was not able to find the ``target_column_name``, i.e. the column containing the target
values (``work_home_actual``) in our case.

-----

The :func:`nannyml.metadata.extraction.extract_metadata` function uses some simple heuristics to yield its results.
You can read more on the inner workings of this function in the :ref:`how it works section <deep_dive_metadata_extraction>`
This means that in some cases it will not succeed in extracting all required information.

The following line of code modifies the :class:`~nannyml.metadata.base.ModelMetadata` object returned by the
:func:`nannyml.metadata.extraction.extract_metadata` function by setting its ``target_column_name`` property.

.. code-block:: python

    >>> metadata.target_column_name = 'work_home_actual'

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

Insights and Follow Ups
-----------------------

.. warning::
    Because the extraction is based on simple rules the results are never guaranteed to be completely correct.
    It is strongly advised to review the results of
    :func:`extract_metadata<nannyml.metadata.extraction.extract_metadata>` and update the values where needed.

    NannyML will raise an :class:`~nannyml.exceptions.MissingMetadataException` when trying to run any functionality
    using incomplete metadata.

.. note::
    We are aware that this boilerplate setup step creates some friction for the end user. We're actively working
    on reducing this friction even more.

To find out more about the columns that should in your dataset, check out the
:ref:`data requirements<data_requirements>` documentation.

You can read the :ref:`how metadata extraction works<deep_dive_metadata_extraction>` to find out more about our
naming conventions and heuristics.

You can put your shiny new metadata to use in :ref:`drift calculation<data-drift>`
or :ref:`performance estimation<performance-estimation>`.
