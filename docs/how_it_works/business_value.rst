.. _business-value-deep-dive:

============================================
Estimation and Calculation of Business Value
============================================

In this page, we will discuss how the ``business_value`` metric works under the hood.
To see how to use the ``business_value`` metric when you have access to the ground truth labels of the data, see the :ref:`business-value-calculation` tutorial.
To see how to use the ``business_value`` metric when you don't have access to the ground truth labels of the data, see the :ref:`business-value-estimation` tutorial.

Intro and Importance of Business Value
--------------------------------------

The ``business_value`` metric offers a way to quantify
the value of a model to a business in terms of the
business's own metrics. At the core, if the value (or cost) of each 
cell in the confusion matrix is known, then the business value of a
model can be calculated.

For example, if the value of a true positive is $100,000, the value of a
true negative is $0, the value of a false positive is $1,000, and
the value of a false negative is $10,000, then the business value of a
can be calculated as follows:

.. math::

    \text{business value} = 100,000 \times \text{number of true positives} + 0 \times \text{number of true negatives} \\
    + 1,000 \times \text{number of false positives} + 10,000 \times \text{number of false negatives}

Calculation of Business Value For Binary Classification
-------------------------------------------------------

More formally, the business value of a model is calculated as follows:

.. math::

    \text{business value} = \sum_{i=1}^{n} \sum_{j=1}^{n} \text{value}_{i,j} \times \text{confusion_matrix}_{i,j}

where :math:`\text{value}_{i,j}` is the business value of a cell in the confusion matrix, and :math:`\text{confusion_matrix}_{i,j}` is the count of observations
in that cell of the confusion matrix.

Since we are in the binary classification case, :math:`n=2`, and the confusion matrix is:

.. math::

    \begin{bmatrix}
    \text{# of true positives} & \text{# of false positives} \\
    \text{# of false negatives} & \text{# of true negatives}
    \end{bmatrix}

And the value matrix is:

.. math::

    \begin{bmatrix}
    \text{value of a true positive} & \text{value of a false positive} \\
    \text{value of a false negative} & \text{value of a true negative}
    \end{bmatrix}

The business value of a binary classification model can thus be generally expressed as:

.. math::

    \text{business value} = \text{value of a true positive} \times \text{# of true positives} \\
    + \text{value of a false positive} \times \text{# of false positives} \\
    + \text{value of a false negative} \times \text{# of false negatives} \\
    + \text{value of a true negative} \times \text{# of true negatives}

Estimation of Business Value For Binary Classification
------------------------------------------------------
In cases where ground truth labels of the data are unavailable, we can still estimate the business value of a model. This is done by using the
:term:`CBPE (Confidence-Based Performance Estimation)` algorithm to estimate the confusion matrix, and then using the business value formula above to calculate
the business value. To read more about the :term:`CBPE (Confidence-Based Performance Estimation)` algorithm, see our :ref:`performance estimation deep dive<performance-estimation-deep-dive>`.

With the estimate of the confusion matrix, we can then estimate the business value of a model using the business value calculation formula above.

Normalization
-------------

The ``business_value`` metric can be normalized so that the value returned is the business value per prediction.
The advantage of this is that it allows for easy comparison of the business value of different models, even if they have
different numbers of predictions. Further, it allows for easy comparison of the business value of the same model on different 
chunks of data, if they have different numbers of predictions as is often the case when using period-based chunking.
To see how to normalize the ``business_value`` metric, see the :ref:`business-value-estimation` tutorial.

Under the hood normalization is quite simple. The total ``business_value`` metric is calculated or estimated as described above, 
and then divided by the number of predictions in a given chunk.
