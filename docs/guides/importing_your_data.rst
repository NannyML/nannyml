.. _import-data:

===========================
Importing data into NannyML
===========================

TLDR
=====

1.  NannyML needs to understand what the inputs and outputs of your ML model look like.
2.  The actual *data* you provide to NannyML are the feature values given to your model
    and the predictions it makes for them.
3.  The ``ModelMetadata`` class stores all this information and uses it in calculations.
4.  You can construct this metadata or extract it based on a sample of your model inputs/outputs.
5.  For metadata extraction to work optimally there are some conventions to follow.

Introduction
============

In this guide you'll learn how to setup NannyML to work on your data. We've provided data for a fictional model: the
**work_from_home** model. It is a binary classifier trying to predict whether someone will be working from home on
a given day or not.

The dataset is provided by the NannyML package. You can see below how to
import it and explore it:

A quick exploration of the dataset:

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
To do this it needs the following information, the monitored model's :term:`Model inputs`,
:term:`Model predictions` and :term:`Target`. This information over time allows
NannyML to track the monitored model's performance and raise any issues that arise.

In order to apply the correct analysis on each of your model inputs NannyML needs some
additional information, such as the kind of data that a feature might hold
(continuous, categorical, ordinal, ...) and the time the prediction was made.
The set of this describing information is what we call :term:`Model metadata`.

.. image:: https://via.placeholder.com/900x300.png?text=model+invocation+process

..
    TODO: insert illustration showing model invocation and assigning names to everything

Combining model inputs and outputs with metadata results in a set of rows called
**observations**.
NannyML will consume a ``pandas.DataFrame`` of these **observations** and turn these into drift
and performance metrics.

.. image:: https://via.placeholder.com/900x300.png?text=annotated+tabular+data


..
    TODO: insert illustration that shows all data in tabular form with annotations


This means that for our example case:

- The **model inputs** are the following columns: ``nationality``, ``dist_to_office``, ``day_of_the_week``,
  ``work_home_yest``, ``transport_mode``, ``industry`` and ``date`` with ``dist_to_office`` and ``date`` being
  continuous features and all others categorical.
- The **model score** is in the ``work_home_pred`` column.
- The **target** is in the ``work_home_actual`` column.
- The ``identifier`` column has a unique value for each row and thus seems like a **metadata** column.


Data requirements
=================

The data provided to NannyML should contain the following *metadata*:

.. list-table::
    :widths: 30, 70
    :header-rows: 1

    * - Name
      - Format
    * - **Identifier**
      - ``Any``
    * - **Timestamp**
      - - *ISO 8601*, e.g. ``2021-10-13T08:47:23Z``
        - *Unix-epoch* in units of seconds, e.g. ``1513393355``
    * - **Model score**
      - ``Float``
    * - **Target**
      - ``1`` when positive, ``0`` otherwise


.. warning::
    The **model score** should contain the actual score or the predicted probability,
    not the predicted label (e.g. ``0`` or ``1``).

.. note::
    **Target** values are only required in the reference data. Performance will be *calculated* using them.
    In the *analysis data* where they are not required, performance can be *estimated*.

Looking back at our example model we can make the following mapping:

.. list-table::
    :widths: 50, 50
    :header-rows: 1

    * - Metadata property
      - Column name
    * - **Identifier**
      - ``identifier``
    * - **Timestamp**
      - ``date``
    * - **Model score**
      - ``work_home_pred``
    * - **Target**
      - ``work_home_actual``

In case any of these metadata columns were missing, you could enrich your dataset using derived or external data
to acquire the necessary columns.

Providing metadata
==================

Preprocessing the example data
------------------------------

After all of this exploratory work it's time to fire up NannyML. We'll tell NannyML to read the example data and
start preprocessing it.

The result of the ``nannyml.preprocess`` function is a tuple. The first element is a ``ModelMetadata``
object that holds the metadata information about the model and its features.

The second returned element is a copy of the uploaded data with some additional columns. NannyML has added copies of
metadata columns and other calculated values such as calibrated model scores. These were given fixed names,
starting with the ``NML_`` prefix. These columns are used by NannyML internally during further processing.

.. code-block:: python

    >>> metadata, prepped_data = nml.preprocess(data=df, model_name='work_from_home')

    NannyML extracted 7 categorical features. Please review these to determine if they should be marked as ordinal instead.

    metadata is still missing values for ['prediction_column_name', 'ground_truth_column_name', 'partition_column_name'].
    Please rectify by renaming columns following automated extraction conventions
    and re-running preprocessing or set metadata properties manually.

    See https://docs.nannyml.com/metadata-extraction for more information

NannyML is warning is about missing metadata values. We can take a closer look at the (partial) metadata that
preprocessing has returned. Also note that the value for the second return variable, ``prepped_data`` is ``None``.

.. code-block:: python

    >>> print(metadata)
    Metadata for model work_from_home

    # Warning - unable to identify all essential data
    # Please identify column names for all '~ UNKNOWN ~' values

    Model problem             binary_classification

    Identifier column         identifier
    Timestamp column          date
    Model score column        ~ UNKNOWN ~
    Target column             ~ UNKNOWN ~

    Features

    Name                 Column               Type            Description
    nationality          nationality          categorical     extracted feature: nationality
    dist_to_office       dist_to_office       continuous      extracted feature: dist_to_office
    day_of_the_week      day_of_the_week      categorical     extracted feature: day_of_the_week
    work_home_yest       work_home_yest       categorical     extracted feature: work_home_yest
    transport_mode       transport_mode       categorical     extracted feature: transport_mode
    work_home_pred       work_home_pred       categorical     extracted feature: work_home_pred
    work_home_actual     work_home_actual     categorical     extracted feature: work_home_actual
    industry             industry             categorical     extracted feature: industry

    >>> prepped_data

Completing the metadata
-----------------------

We'll complete the model metadata by providing the missing values. Since there is no column containing the partition
of the data, we'll have to add one to the data manually first.

.. code-block:: python

    >>> metadata.prediction_column_name = 'work_home_pred'
    >>> metadata.ground_truth_column_name = 'work_home_actual'
    >>> df['partition'] = 'reference'
    >>> metadata.partition_column_name = 'partition']
    >>> metadata.is_complete()
    (True, [])  # yay, our metadata is all good to go!

We can now re-run the preprocessing step. Mind the added parameter to the :code:`preprocess` call.
It allows us to provide our completed metadata and will no longer try to extract it from our data.
We see that :code:`prepped_data` has been populated now and it contains some additional technical columns.

.. code-block:: python

    >>> metadata, prepped_data = nml.preprocess(data=df, model_name='work_from_home', model_metadata=metadata)
    >>> prepped_data
           identifier nationality  dist_to_office  day_of_the_week  ...  nml_meta_prediction nml_meta_ground_truth  nml_meta_partition  nml_calibrated_score
    0           27639          FR      467.420495                7  ...                    1                     1           reference              0.501924
    1           62250          BE      446.879715                6  ...                    1                     1           reference              0.501924
    2            7140          BE      228.347808                2  ...                    1                     0           reference              0.501924
    3           44561          BE      375.441565                5  ...                    1                     0           reference              0.501924
    4           92985          FR      286.660112                4  ...                    0                     1           reference              0.499669
    ...           ...         ...             ...              ...  ...                  ...                   ...                 ...                   ...
    99995       53343          BE      303.228327                3  ...                    1                     0           reference              0.501924
    99996       52819          BE      131.051512                2  ...                    0                     0           reference              0.499669
    99997       79445          BE      135.600934                1  ...                    1                     0           reference              0.501924
    99998       58108          BE      431.322066                5  ...                    1                     1           reference              0.501924
    99999       57400          BE      453.808103                7  ...                    0                     1           reference              0.499669

Loading additional data
========================

We can now reuse the existing metadata to preprocess the analysis data as well. As with the reference data,
the partition column is still lacking so we'll add that one first.

.. code-block:: python

    >>> df_analysis = pd.read_csv('../data/work_from_home_analysis.csv')
    >>> df_analysis['partition'] = 'analysis'
    >>> _, prepped_data_analysis = nml.preprocess(df_analysis, 'work_from_home', metadata)

And that's it! Both datasets are ready to use. Check out the next guide on how to calculate drift!
