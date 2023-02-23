Take a machine learning model that uses some multidimensional input data
:math:`\mathbf{X}` and makes predictions :math:`y`.

The model has been trained on some data distribution :math:`P(\mathbf{X})`.
There is data drift when the production data comes from a different distribution
:math:`P(\mathbf{X'}) \neq P(\mathbf{X})`.

A machine learning model operating on an input distribution different from
the one it has been trained on may underperform. It is crucial to detect
data drift in a timely manner when a model is in production.

By investigating the characteristics of an observed drift, the causes
of any performance change can be identified.

There is also a special case of data drift called label shift. In this case, the outcome
distributions between the training and production data are different, meaning
:math:`P(y') \neq P(y)`. However, the relationship between the population characteristics and
a specific outcome does not change, namely :math:`P(\mathbf{X'}|y') = P(\mathbf{X}|y)`.

Data drift is one of the two main reason for silent model failure.
The second one is concept drift, where the relationship between the model inputs and the target changes.
In this case we have: :math:`P(y'|\mathbf{X'}) \neq P(y|\mathbf{X})`.
In production it can happen that a model experiences data drift and concept drift simultaneously.

Below we can explore the various ways in which NannyML allows you to detect data drift.
