.. _dataset-synthetic-multiclass:

===========================================
Synthetic Multiclass Classification Dataset
===========================================

NannyML provides a synthetic dataset describing a multiclass classification problem,
to make it easier to test and document its features.

To find out what requirements NannyML has for datasets, check out :ref:`Data Requirements<data_requirements>`.

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


+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+---------------------+-----------------------------+--------------------------------+------------------------------+--------------+---------------+
|    | acq_channel   |   app_behavioral_score |   requested_credit_limit | app_channel   |   credit_bureau_score |   stated_income | is_customer   | timestamp           |   y_pred_proba_prepaid_card |   y_pred_proba_highstreet_card |   y_pred_proba_upmarket_card | y_pred       | y_true        |
+====+===============+========================+==========================+===============+=======================+=================+===============+=====================+=============================+================================+==============================+==============+===============+
|  0 | Partner3      |               1.80823  |                      350 | web           |                   309 |           15000 | True          | 2020-05-02 02:01:30 |                        0.97 |                           0.03 |                         0    | prepaid_card | prepaid_card  |
+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+---------------------+-----------------------------+--------------------------------+------------------------------+--------------+---------------+
|  1 | Partner2      |               4.38257  |                      500 | mobile        |                   418 |           23000 | True          | 2020-05-02 02:03:33 |                        0.87 |                           0.13 |                         0    | prepaid_card | prepaid_card  |
+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+---------------------+-----------------------------+--------------------------------+------------------------------+--------------+---------------+
|  2 | Partner2      |              -0.787575 |                      400 | web           |                   507 |           24000 | False         | 2020-05-02 02:04:49 |                        0.47 |                           0.35 |                         0.18 | prepaid_card | upmarket_card |
+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+---------------------+-----------------------------+--------------------------------+------------------------------+--------------+---------------+

The model uses 7 features:

- **acq_channel** - a categorical feature with 5 categories describing the acquisition channel for the new customer.
  Organic refers to customers brought on by the company whereas Partner1-4 refers to customers brought on by
  outside partners.
- **app_behavioral_score** - a numerical feature. This score is determined by characteristics derived from how the
  new customer filled in and submitted their application.
- **requested_credit_limit** - a numerical feature. The credit limit the customer selected as appropriate for their
  needs.
- **app_channel** - a categorical feature with 3 categories describing how the application was submitted. It can
  be in-store, from the website or from a mobile device.
- **credit_bureau_score** - a numerical feature. The credit score provided by the credit bureau that assesses the credit
  worthiness of the new customer. The higher the score the more credit-worthy the customer.
- **stated_income** - a numerical feature. The yearly income of the customer, as stated by them.
- **is_customer** - a categorical feature with 2 categories describing whether the new customer has an existing
  relationship with the business.

The model predicts a probability for all classes with the **y_pred_proba_prepaid_card**,
**y_pred_proba_highstreet_card**, **y_pred_proba_upmarket_card** columns.
A class prediction is also available from the **y_pred** column. The **y_true** is the :term:`Target` column
with the most appropriate product choice for a given customer.


There is also an auxiliary column that is helpful but not used by the monitored model:

- **timestamp** - a date column informing us of the date the prediction was made.
