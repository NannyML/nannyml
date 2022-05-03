.. _data-drift:

====================
Detecting Data Drift
====================

Take a machine learning model that uses some multidimensional input data
:math:`\mathbf{X}` and makes predictions :math:`y`.

The model has been trained on some data distribution :math:`P(\mathbf{X})`.
There is data drift when the production data comes from a different distribution
:math:`P(\mathbf{X'}) \neq P(\mathbf{X})`.

A machine learning model operating on an input distribution different from
the one it has been trained on will probably underperform. It is therefore crucial to detect
data drift, in a timely manner, when a model is in production. By further investigating the
characteristics of the observed drift, the data scientists operating the model
can potentially estimate the impact of the drift on the model's performance.

There is also a special case of data drift called label shift. In this case, the outcome
distributions between the training and production data are different, meaning
:math:`P(y') \neq P(y)`. However, the relationship between the population characteristics and
a specific outcome does not change, namely :math:`P(\mathbf{X'}|y') = P(\mathbf{X}|y)`.

It is important to note that data drift is not the only change that can happen when there is a
machine learning model in production. Another important change is concept drift, where the relationship
between the model inputs and the target changes. In this case we have: :math:`P(y'|\mathbf{X'}) \neq P(y|\mathbf{X})`.
In production it can happen that a model experiences data drift and concept drift simultaneously.


Below we see the various ways in which NannyML detects data drift.

.. toctree::
   :maxdepth: 2

   detecting_data_drift/univariate_drift_detection
   detecting_data_drift/multivariate_drift_detection
   detecting_data_drift/drift_detection_for_model_outputs
   detecting_data_drift/drift_detection_for_model_targets
