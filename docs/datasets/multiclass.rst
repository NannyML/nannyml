.. _dataset-synthetic-multiclass:

===========================================
Synthetic Multiclass Classification Dataset
===========================================

NannyML provides a synthetic dataset describing a multiclass classification problem, 
to make it easier to test and document its features.

Problem Description
===================

The dataset describes a machine learning model that tries to predict the most appropriate product
for new customers applying for a credit card. There are three options. Prepaid cards with low
credit limits, highstreet cards with low credit limits and high interest rates, and upmarket cards
with higher credit limits and lower interest rates.

Dataset Description
===================

A sample of the dataset can be seen below.


.. code-block:: python

    >>> import nannyml as nml
    >>> reference, analysis, analysis_targets = nml.datasets.load_synthetic_binary_classification_dataset()
    >>> display(reference.head(3))

+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-------------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+--------------+---------------+
|    | acq_channel   |   app_behavioral_score |   requested_credit_limit | app_channel   |   credit_bureau_score |   stated_income | is_customer   | partition   |   identifier | timestamp           |   y_pred_proba_prepaid_card |   y_pred_proba_highstreet_card |   y_pred_proba_upmarket_card | y_pred       | y_true        |
+====+===============+========================+==========================+===============+=======================+=================+===============+=============+==============+=====================+=============================+================================+==============================+==============+===============+
|  0 | Partner3      |               1.80823  |                      350 | web           |                   309 |           15000 | True          | reference   |        60000 | 2020-05-02 02:01:30 |                        0.97 |                           0.03 |                         0    | prepaid_card | prepaid_card  |
+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-------------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+--------------+---------------+
|  1 | Partner2      |               4.38257  |                      500 | mobile        |                   418 |           23000 | True          | reference   |        60001 | 2020-05-02 02:03:33 |                        0.87 |                           0.13 |                         0    | prepaid_card | prepaid_card  |
+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-------------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+--------------+---------------+
|  2 | Partner2      |              -0.787575 |                      400 | web           |                   507 |           24000 | False         | reference   |        60002 | 2020-05-02 02:04:49 |                        0.47 |                           0.35 |                         0.18 | prepaid_card | upmarket_card |
+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-------------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+--------------+---------------+


The model uses 7 features:

- `acq_channel`: A categorical feature with 5 categories describing the acquisition channel for the new customer.
  Organic refers to customers brought on by the company whereas Partner1-4 refers to customers brought on by
  outside partners.
- `app_behavioral_score`:  A numerical feature. This score is determined by characteristics derived from how the
  new customer filled in and submitted their application.
- `requested_credit_limit`: A numerical feature. The credit limit the customer selected as appropriate for their
  needs.
- `app_channel`: A categorical feature with 3 categories describing how the application was submitted. It can
  be in-store, from the website or from a mobile device.
- `credit_bureau_score`: A numerical feature. The credit score provided by the credit bureau that assesses the credit
  worthiness of the new customer. The higher the score the more credit-worthy the customer.
- `stated_income`: A numerical feature. The yearly income of the customer, as stated by them.
- `is_customer`:  A categorical feature with 2 categories describing whether the new customer has an existing
  relationship with the business.

The model predicts a probability for all classes with the `y_pred_proba_prepaid_card`,
`y_pred_proba_highstreet_card`, `y_pred_proba_upmarket_card` columns.
A class prediction is also available from the `y_pred` column. The `y_true` is the :term:`Target` column
with the most appropriate product choice for a given customer.


There are also three auxiliarry columns that are helpful but not used by the monitored model:

- `identifier`: A unique number referencing each new customer. This is very useful for joining the target
  results on the analysis dataset, when we want to :ref:`compare estimated with realized performace.<compare_estimated_and_realized_performance>`.
- `timestamp`: A date column informing us of the date the prediction was made.
- `partition`: The partition column tells us which :term:`Data Period` each row comes from.


Metadata Extraction
===================

The dataset's columns are named so that the heuristics NannyML uses to extract metadata can
identify them. We can see below how to extract metadata.


.. code-block:: python

    >>> metadata = nml.extract_metadata(
    ...     data = reference,
    ...     model_name='credit_card_segment',
    ...     model_type='classification_binary',
    ...     exclude_columns=['identifier']
    >>> )
    >>> metadata.is_complete()

We can now see all the metadata that NannyML has inferred about the model.

.. code-block:: python

    >>> metadata.to_df()

+----+-----------------------------------+------------------------------+-------------+-------------------------------------------------+
|    | label                             | column_name                  | type        | description                                     |
+====+===================================+==============================+=============+=================================================+
|  0 | timestamp_column_name             | timestamp                    | continuous  | timestamp                                       |
+----+-----------------------------------+------------------------------+-------------+-------------------------------------------------+
|  1 | partition_column_name             | partition                    | categorical | partition                                       |
+----+-----------------------------------+------------------------------+-------------+-------------------------------------------------+
|  2 | target_column_name                | y_true                       | categorical | target                                          |
+----+-----------------------------------+------------------------------+-------------+-------------------------------------------------+
|  3 | acq_channel                       | acq_channel                  | categorical | extracted feature: acq_channel                  |
+----+-----------------------------------+------------------------------+-------------+-------------------------------------------------+
|  4 | app_behavioral_score              | app_behavioral_score         | continuous  | extracted feature: app_behavioral_score         |
+----+-----------------------------------+------------------------------+-------------+-------------------------------------------------+
|  5 | requested_credit_limit            | requested_credit_limit       | categorical | extracted feature: requested_credit_limit       |
+----+-----------------------------------+------------------------------+-------------+-------------------------------------------------+
|  6 | app_channel                       | app_channel                  | categorical | extracted feature: app_channel                  |
+----+-----------------------------------+------------------------------+-------------+-------------------------------------------------+
|  7 | credit_bureau_score               | credit_bureau_score          | continuous  | extracted feature: credit_bureau_score          |
+----+-----------------------------------+------------------------------+-------------+-------------------------------------------------+
|  8 | stated_income                     | stated_income                | categorical | extracted feature: stated_income                |
+----+-----------------------------------+------------------------------+-------------+-------------------------------------------------+
|  9 | is_customer                       | is_customer                  | categorical | extracted feature: is_customer                  |
+----+-----------------------------------+------------------------------+-------------+-------------------------------------------------+
| 10 | y_pred_proba_prepaid_card         | y_pred_proba_prepaid_card    | continuous  | extracted feature: y_pred_proba_prepaid_card    |
+----+-----------------------------------+------------------------------+-------------+-------------------------------------------------+
| 11 | y_pred_proba_highstreet_card      | y_pred_proba_highstreet_card | continuous  | extracted feature: y_pred_proba_highstreet_card |
+----+-----------------------------------+------------------------------+-------------+-------------------------------------------------+
| 12 | y_pred_proba_upmarket_card        | y_pred_proba_upmarket_card   | continuous  | extracted feature: y_pred_proba_upmarket_card   |
+----+-----------------------------------+------------------------------+-------------+-------------------------------------------------+
| 13 | prediction_column_name            | y_pred                       | continuous  | predicted label                                 |
+----+-----------------------------------+------------------------------+-------------+-------------------------------------------------+
| 14 | predicted_probability_column_name |                              | continuous  | predicted score/probability                     |
+----+-----------------------------------+------------------------------+-------------+-------------------------------------------------+

For more information about specifying metadata look at :ref:`Providing Metadata<import-data>`.
