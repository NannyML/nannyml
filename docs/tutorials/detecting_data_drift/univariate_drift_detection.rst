.. _univariate_drift_detection:

==========================
Univariate Drift Detection
==========================


Just The Code
-------------

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cells: 1 3 4 10 12 14 16

.. _univariate_drift_detection_walkthrough:

.. admonition:: **Advanced configuration**
    :class: hint

    - To learn how :class:`~nannyml.chunk.Chunk` works and to set up custom chunkings check out the :ref:`chunking tutorial <chunking>`
    - To learn how :class:`~nannyml.thresholds.ConstantThreshold` works and to set up custom threshold check out the :ref:`thresholds tutorial <thresholds>`

Walkthrough
-----------

NannyML's univariate approach for data drift looks at each variable individually and compares the
:ref:`chunks<chunking>` created from the analysis :ref:`data period<data-drift-periods>` with the reference period.
You can read more about periods and other data requirements in our section on :ref:`data periods<data-drift-periods>`.

The comparison results in a single number, a drift metric, representing the amount of drift between the reference and
analysis chunks. NannyML calculates them for every chunk, allowing you to track them over time.

NannyML offers both statistical tests as well as distance measures to detect drift. They are referred to as
`methods`. Some methods only apply to continuous data, others to categorical data and some might be used on both.
NannyML lets you choose which methods to use for these two types of data.



We begin by loading some synthetic data provided in the NannyML package. This is data for a binary classification model, but other model types operate in the same way.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cells: 1

.. nbtable::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cell: 2

The :class:`~nannyml.drift.univariate.calculator.UnivariateDriftCalculator` class implements the functionality needed for univariate drift detection.
First, we need to instantiate it with the appropriate parameters:

- The names of the columns to be evaluated.
- A list of methods to use on continuous columns. You can chose from :ref:`kolmogorov_smirnov<univ_cont_method_ks>`,
  :ref:`jensen_shannon<univariate-drift-detection-cont-jensen-shannon>`, :ref:`wasserstein<univariate-drift-detection-cont-wasserstein>`
  and :ref:`hellinger<univariate-drift-detection-cont-hellinger>`.
- A list of methods to use on categorical columns. You can choose from :ref:`chi2<univ_cat_method_chi2>`, :ref:`jensen_shannon<univ_cat_method_js>`,
  :ref:`l_infinity<univ_cat_method_l8>`, and :ref:`hellinger<univ_cat_method_hellinger>`.
- Optionally, the name of the column containing the observation timestamps.
- Optionally, a chunking approach or a predefined chunker. If neither is provided, the default chunker creating 10 chunks will be used.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cells: 3

Next, the :meth:`~nannyml.base.AbstractCalculator.fit` method needs
to be called on the reference data, which provides the baseline that the analysis data will be compared with. Then the
:meth:`~nannyml.base.AbstractCalculator.calculate` method will
calculate the drift results on the provided data.

The results can be filtered to only include a certain data period, method, or column by using the :meth:`~nannyml.base.AbstractResult.filter` method.
You can evaluate the result data by converting the results into a `DataFrame`,
by calling the :meth:`~nannyml.base.AbstractResult.to_df` method.
By default, this will return a `DataFrame` with a multi-level index. The first level represents the column, the second level
is the method and the third level is the values, thresholds, and alerts for that method.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cells: 4

.. nbtable::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cell: 5

You can also disable the multi-level index behavior and return a flat structure by setting ``multilevel=False``.
Both the `column name` and the `method` have now been included within the column names.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cells: 6

.. nbtable::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cell: 7


The drift results from the reference data are accessible though the :meth:`~nannyml.base.AbstractResult.filter` method of the drift calculator results:

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cells: 8

.. nbtable::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cell: 9

The next step is visualizing the results. NannyML can plot both the `drift` and `distribution` for a given column.
We will first plot the ``jensen_shannon`` method results for each continuous column shown below.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cells: 10

.. _univariate_drift_detection_tenure:
.. image:: /_static/tutorials/detecting_data_drift/univariate_drift_detection/jensen-shannon-continuous.svg

Note that among the columns shown ``y_pred_proba`` is included. This means that the drift calculator is not only limited to model features,
but can also be applied to model scores and predictions.
This also applies to categorical columns. The plot below shows the ``chi2`` results for each categorical column
and that also includes the ``y_pred`` column.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cells: 12

.. image:: /_static/tutorials/detecting_data_drift/univariate_drift_detection/shi-2-categorical.svg



NannyML also shows details about the distributions of continuous and categorical variables.

For continuous variables, NannyML plots the estimated probability distribution of the variable for
each chunk in a plot called joyplot. The chunks where the drift was detected are highlighted.

Using the code below, we can create joyplots for the model’s continuous variables.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cells: 14

.. image:: /_static/tutorials/detecting_data_drift/univariate_drift_detection/joyplot-continuous.svg

NannyML plots stacked bar charts for categorical variables to show the variable's distribution for each chunk.
If a variable has more than 5 categories, the top 4 are displayed and the rest are grouped together to make
the plots easier to view. In addition, the chunks where drift was detected are highlighted.

We can create stacked bar charts for the model's categorical variables with
the code below.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cells: 16

.. image:: /_static/tutorials/detecting_data_drift/univariate_drift_detection/stacked-categorical.svg

Insights
--------

After reviewing the above results we have a good understanding of what has changed in our
model's population.


What Next
---------

The :ref:`Performance Estimation<performance-estimation>` functionality of NannyML can help provide estimates of the impact of the
observed changes to Model Performance. The :ref:`ranking<tutorial-ranking>` functionality can help rank drifted features to
suggest which ones to prioritize for further investigation if needed. This would be an ad-hoc investigating that is not covered by NannyML.
