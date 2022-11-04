.. _univariate_drift_detection:

======================================
Univariate Statistical Drift Detection
======================================


Just The Code
-------------

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cells: 1 3 4 10 12 14 16

.. _univariate_drift_detection_walkthrough:

Walkthrough
-----------

NannyML's univariate approach for data drift looks at each variable individually and compares the
:ref:`chunks<chunking>` created from the analysis :ref:`data period<data-drift-periods>` with the reference period.
You can read more about periods and other data requirements in our section on :ref:`data periods<data-drift-periods>`

The comparison results in a single number, a drift metric, representing the amount of drift between the reference and
analysis chunks. NannyML calculates them for every chunk, allowing you to track them over time.

NannyML offers both statistical tests as well as distance measures to detect drift. They are being referred to as
`methods`. Some methods are only applicable to continuous data, others to categorical data and some might be used on both.
NannyML lets you choose which methods are to be used on these two types of data.

We begin by loading some synthetic data provided in the NannyML package. This is data for a binary classification model, but other model types operate in the same way.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cells: 1

.. nbtable::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cell: 2

The :class:`~nannyml.drift.univariate.calculator.UnivariateDriftCalculator` class implements the functionality needed for univariate drift detection.
We need to instantiate it with appropriate parameters:
-  the names of the columns to be evaluated
-  the name of a column containing the observation timestamps, optional
-  a list of methods to use on continuous columns
-  a list of methods to use on categorical columns
-  some specifications on how to chunk the data

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cells: 3

Next, the :meth:`~nannyml.drift.univariate.calculator.UnivariateDriftCalculator.fit` method needs
to be called on the reference data, which provides the baseline that the analysis data will be compared with. Then the
:meth:`~nannyml.drift.univariate.calculator.UnivariateDriftCalculator.calculate` method will
calculate the drift results on the data provided to it.

The results can be filtered to only include a certain data period, method or column by using the ``filter`` method.
You can evaluate the result data by converting the results into a `DataFrame`,
by calling the :meth:`~nannyml.drift.univariate.result.Result.to_df` method.
By default this will return a `DataFrame` with a multi-level index. The first level represents the column, the second level
is the method that was used and the third level are the values, thresholds and alerts for that method.

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


The drift results from the reference data are accessible though the ``filter()`` method of the drift calculator results:

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cells: 8

.. nbtable::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cell: 9

The next step is visualizing the results. NannyML can plot both the `drift` as well as `distribution` for a given column.
We'll first plot the ``jensen_shannon`` method results for each continuous column:

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cells: 10

.. image:: /_static/drift-guide-distance_from_office.svg

.. image:: /_static/drift-guide-gas_price_per_litre.svg

.. _univariate_drift_detection_tenure:
.. image:: /_static/drift-guide-tenure.svg

.. image:: /_static/drift-guide-public_transportation_cost.svg

We then plot the ``chi2`` results for each categorical column:

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cells: 12

.. image:: /_static/drift-guide-wfh_prev_workday.svg

.. image:: /_static/drift-guide-workday.svg

.. image:: /_static/drift-guide-salary_range.svg


NannyML also shows details about the distributions of continuous and categorical variables.

For continuous variables NannyML plots the estimated probability distribution of the variable for
each chunk in a plot called joyplot. The chunks where drift was detected are highlighted.
We can create joyplots for the model's continuous variables as following:

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cells: 14

.. image:: /_static/drift-guide-joyplot-distance_from_office.svg

.. image:: /_static/drift-guide-joyplot-gas_price_per_litre.svg

.. image:: /_static/drift-guide-joyplot-public_transportation_cost.svg

.. image:: /_static/drift-guide-joyplot-tenure.svg

For categorical variables NannyML plots stacked bar charts to show the variable's distribution for each chunk.
If a variable has more than 5 categories, the top 4 are displayed and the rest are grouped together to make
the plots easier to view. We can stacked bar charts for the model's categorical variables with
the code below:

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cells: 16

.. image:: /_static/drift-guide-stacked-salary_range.svg

.. image:: /_static/drift-guide-stacked-wfh_prev_workday.svg

.. image:: /_static/drift-guide-stacked-workday.svg

NannyML can also rank features according to how many alerts they have had for all methods.
By setting the ``only_drifting`` parameter you can view the ranking of either all model inputs, or just the drifting ones.
NannyML provides a dataframe with the resulting ranking of features.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cells: 18

.. nbtable::
    :path: ./example_notebooks/Tutorial - Drift - Univariate.ipynb
    :cell: 19

Insights
--------

After reviewing the above results we have a good understanding of what has changed in our
model's population.

What Next
---------

The :ref:`Performance Estimation<performance-estimation>` functionality of NannyML can help provide estimates of the impact of the
observed changes to Model Performance.

If needed, we can investigate further as to why our population characteristics have
changed the way they did. This is an ad-hoc investigating that is not covered by NannyML.
