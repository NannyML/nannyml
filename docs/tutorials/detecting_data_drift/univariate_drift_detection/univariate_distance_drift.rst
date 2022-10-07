.. _univariate_distance_drift_detection:

======================================
Univariate Distance Drift Detection
======================================




Just The Code
-------------

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Inputs - Univariate Distance.ipynb
    :cells: 1 3 4 6 8 10 12 14

.. _univariate_distance_drift_detection_walkthrough:

Walkthrough
-----------

NannyML's Distance Univariate approach for data drift looks at each variable individually and measures the distance between the distributions
of each feature comparing the
:ref:`chunks<chunking>` created from the analysis :ref:`data period<data-drift-periods>` with the reference period.
You can read more about the data required in our section on :ref:`data periods<data-drift-periods>`

NannyML uses the :term:`Jensen-Shannon divergence<Jensen-Shannon Divergencet>` fwhich works for both continuous features and categorical features.
It takes values between 0 and 1.
Both tests provide a statistic where they measure
the observed drift and a p-value that shows how likely we are to get the observed sample under the assumption that there was no drift.

If the p-value is less than 0.05 NannyML considers the result unlikely to be due to chance and issues an alert for the associated chunk and feature.

We begin by loading some synthetic data provided in the NannyML package. This is data for a binary classification model, but other model types operate in the same way.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Inputs - Univariate Statistical.ipynb
    :cells: 1

.. nbtable::
    :path: ./example_notebooks/Tutorial - Drift - Model Inputs - Univariate Statistical.ipynb
    :cell: 2

The :class:`~nannyml.drift.model_inputs.univariate.statistical.calculator.UnivariateStatisticalDriftCalculator`
class implements the functionality needed for Univariate Drift Detection. We need to instantiate it with appropriate parameters - the column headers of the features that we want to run drift detection on, and the timestamp column header. The features can be passed in as a simple list of strings, but here we have created this list by excluding the columns in the dataframe that are not features, and passed that into the argument.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Inputs - Univariate Statistical.ipynb
    :cells: 3

Next, the :meth:`~nannyml.drift.model_inputs.univariate.statistical.calculator.UnivariateStatisticalDriftCalculator.fit` method needs
to be called on the reference data, which provides the baseline that the analysis data will be compared with. Then the
:meth:`~nannyml.drift.model_inputs.univariate.statistical.calculator.UnivariateStatisticalDriftCalculator.calculate` method will
calculate the drift results on the data provided to it.

We then display a small subset of our results by specifying columns in the
:meth:`~nannyml.drift.model_inputs.univariate.statistical.calculator.UnivariateStatisticalDriftCalculator.calculate.results` method.

NannyML returns a dataframe with 3 columns for each feature. The first column contains the corresponding test
statistic. The second column contains the corresponding p-value and the third column says whether there
is a drift alert for that feature and chunk.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Inputs - Univariate Statistical.ipynb
    :cells: 4

.. nbtable::
    :path: ./example_notebooks/Tutorial - Drift - Model Inputs - Univariate Statistical.ipynb
    :cell: 5

The drift results from the reference data are accessible though the ``previous_reference_results`` property of the drift calculator:

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Inputs - Univariate Statistical.ipynb
    :cells: 6

.. nbtable::
    :path: ./example_notebooks/Tutorial - Drift - Model Inputs - Univariate Statistical.ipynb
    :cell: 7

NannyML can also visualize those results on plots.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Inputs - Univariate Statistical.ipynb
    :cells: 8

.. image:: /_static/drift-guide-distance_from_office.svg

.. image:: /_static/drift-guide-gas_price_per_litre.svg

.. _univariate_distance_drift_detection_tenure:
.. image:: /_static/drift-guide-tenure.svg

.. image:: /_static/drift-guide-wfh_prev_workday.svg

.. image:: /_static/drift-guide-workday.svg

.. image:: /_static/drift-guide-public_transportation_cost.svg

.. image:: /_static/drift-guide-salary_range.svg

NannyML also shows details about the distributions of continuous variables and categorical variables.
For continuous variables NannyML plots the estimated probability distribution of the variable for
each chunk in a plot called joyplot. The chunks where drift was detected are highlighted.
We can create joyplots for the model's continuous variables with
the code below:

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Inputs - Univariate Statistical.ipynb
    :cells: 10

.. image:: /_static/drift-guide-joyplot-distance_from_office.svg

.. image:: /_static/drift-guide-joyplot-gas_price_per_litre.svg

.. image:: /_static/drift-guide-joyplot-public_transportation_cost.svg

.. image:: /_static/drift-guide-joyplot-tenure.svg


NannyML can also plot details about the distributions of different features. In these plots, NannyML highlights the areas with possible data drift.
If we want to focus only on the categorical plots, we can specify that only these be plotted.

For categorical variables NannyML plots stacked bar charts to show the variable's distribution for each chunk.
If a variable has more than 5 categories, the top 4 are displayed and the rest are grouped together to make
the plots easier to view. We can stacked bar charts for the model's categorical variables with
the code below:

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Inputs - Univariate Statistical.ipynb
    :cells: 12

.. image:: /_static/drift-guide-stacked-salary_range.svg

.. image:: /_static/drift-guide-stacked-wfh_prev_workday.svg

.. image:: /_static/drift-guide-stacked-workday.svg

NannyML can also rank features according to how many alerts they have had within the data analyzed
for data drift. NannyML allows viewing the ranking of all the model inputs, or just the ones that have drifted.
NannyML provides a dataframe with the resulting ranking of features.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Inputs - Univariate Statistical.ipynb
    :cells: 14

.. nbtable::
    :path: ./example_notebooks/Tutorial - Drift - Model Inputs - Univariate Statistical.ipynb
    :cell: 15

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
