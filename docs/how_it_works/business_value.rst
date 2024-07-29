.. _business-value-deep-dive:

=========================================
Business Value Estimation and Calculation
=========================================

The **business_value** metric provides a way to tie the performance of a model to
monetary or business oriented outcomes.
In this page, we will discuss how the **business_value** metric works under the hood.

Introduction to Business Value
------------------------------

The **business_value** metric offers a way to quantify the value of a model in terms of the
business's own metrics. At the core, if the business value (or cost) of each
outcome in the :term:`confusion matrix<Confusion Matrix>` is known, then the business value of a
model can either be *calculated* using the :ref:`realized Performance Calculator<performance-calculation>` if
the ground truth labels are available or *estimated* using :ref:`Performance Estimation<performance-estimation>`
if the ground truth labels are not available.

More specifically, we know that each prediction made by a binary classification models
can be one of four outcomes:

- True Positive (TP): The model correctly predicts a positive outcome.
- True Negative (TN): The model correctly predicts a negative outcome.
- False Positive (FP): The model incorrectly predicts a positive outcome.
- False Negative (FN): The model incorrectly predicts a negative outcome.

The business value of each of these four outcomes can be calculated according to actual
business results and costs. The total business value of a model
can then be calculated by summing the business value of each prediction.

For example, if the value of a true positive is $100,000, the value of a
true negative is $0, the value of a false positive is $1,000, and
the value of a false negative is $10,000, then the business value of a
can be calculated as follows:

.. math::

    \text{business value} = 100,000 \times \text{number of true positives} + 0 \times \text{number of true negatives} \\
    + 1,000 \times \text{number of false positives} + 10,000 \times \text{number of false negatives}

Business Value Formula
----------------------

We can formalize the intuition above as follows:

.. math::

    \text{business value} = \sum_{i=1}^{n} \sum_{j=1}^{n} \text{business_value}_{i,j} \times \text{confusion_matrix}_{i,j}

where :math:`\text{business_value}_{i,j}` is the business value of a cell in the
:term:`confusion matrix<Confusion Matrix>`, and :math:`\text{confusion_matrix}_{i,j}` is the count of
observations in that cell of the :term:`confusion matrix<Confusion Matrix>`. Using the confusion 
matrix notation the element on the i-th row and j-column of the business value matrix tells us the value
of the i-th target when we have predicted the j-th value.

.. note::
    In Multiclass classification the classes are ordered alphanumerically.
    This is used in the creation of the confusion matrix. The rows of the confusion matrix
    represent target values in the corresponding alphanumerical order. And the columns
    of the confusion matrix represent predicted classes in the same alphanumerical order.
    Therefore the elements of the business value matrix should be constructed accordingly.

For binary classification this formula is easier to manage hence we will use it as an example. Classificatio problems
with more classes follow the same pattern.
Using the `sklearn confusion matrix convention`_ we designate label 0 as negative and label 1 as positive.
Hence we can write the :term:`confusion matrix<Confusion Matrix>` as:

.. math::

    \begin{bmatrix}
    \text{# of true negatives} & \text{# of false positives} \\
    \text{# of false negatives} & \text{# of true positives}
    \end{bmatrix}

Note that target values are represented by rows and predicted values are represented by columns.
This means that the first row contains values that have resulted in the negative outcome
while the first column contains values that were predicted with negative label.
The correspondings :term:`business value matrix` is:

.. math::

    \begin{bmatrix}
    \text{value of a true negative} & \text{value of a false positive} \\
    \text{value of a false negative} & \text{value of a true positive}
    \end{bmatrix}

The business value of a binary classification model can thus be generally expressed as:

.. math::

    \text{business value} = (\text{value of a true negative}) \cdot (\text{# of true negatives}) \\
    + (\text{value of a false positive}) \cdot (\text{# of false positives}) \\
    + (\text{value of a false negative}) \cdot (\text{# of false negatives}) \\
    + (\text{value of a true positive}) \cdot (\text{# of true positives})

Calculation of Business Value For Classification
------------------------------------------------

When the ground truth labels are available, the business value of a model can be calculated by using the
values from the realized :term:`confusion matrix<Confusion Matrix>`,
and then using the business value formula above to calculate the business value.

For a tutorial on how to calculate the business value of a model,
see our :ref:`business-value-calculation` and :ref:`multiclass-business-value-calculation` tutorials.

Estimation of Business Value For Classification
-----------------------------------------------

In cases where ground truth labels of the data are unavailable, we can still estimate the business value of a model.
This is done by using the :term:`CBPE (Confidence-Based Performance Estimation)` algorithm to estimate the
:term:`confusion matrix<Confusion Matrix>`, and then using the business value formula above to obtain a business value estimate.
To read more about the :term:`CBPE (Confidence-Based Performance Estimation)` algorithm,
see our :ref:`performance estimation deep dive<how-it-works-cbpe>`.

For a tutorial on how to estimate the business value of a model, see our :ref:`business-value-estimation`
and :ref:`multiclasss-business-value-estimation` tutorials.

Normalization
-------------

The **business_value** metric can be normalized so that the value returned is the business value per prediction.
The advantage of this is that it allows for easy comparison of the business value of different models, even if they have
different numbers of predictions. Further, it allows for easy comparison of the business value of the same model on different
chunks of data, if they have different numbers of predictions as is often the case when using period-based chunking.

Under the hood normalization is quite simple. The total **business_value** metric is calculated or estimated as described above,
and then divided by the number of predictions in a given chunk.

Normalization is supported for both estimation and calculation of business value.
Check out the :ref:`business-value-calculation` tutorial and the :ref:`business-value-estimation` tutorial
for examples of how to normalize the business value metric.


.. _`sklearn confusion matrix convention`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
