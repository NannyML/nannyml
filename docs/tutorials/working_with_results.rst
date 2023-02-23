.. _working_with_results:

====================
Working with results
====================

What are NannyML Results?
-------------------------

In NannyML any calculation will return a :class:`Result<nannyml.base.AbstractCalculatorResult>` object. Not returning
a DataFrame directly allows NannyML to separate the concerns of storing calculation results and having users interact
with them. It also means we can  provide some additional useful methods on top of the results, for example filtering
and plotting.

Just the code
-------------

.. nbimport::
    :path: ./example_notebooks/Tutorial - Working with results.ipynb
    :cells: 1 2 3 6 7 9 11

Walkthrough
-----------

In order to obtain results we first have to perform some calculation. We'll start by loading the `reference` and
`analysis` sample data for binary classification.
We'll perform univariate drift detection on a number of columns whose names are printed below.
Knowing the column names will help you understand this walkthrough better.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Working with results.ipynb
    :cells: 1
    :show_output:

We then set up the :class:`~nannyml.drift.univariate.calculator.UnivariateDriftCalculator` by specifying the names
of the columns to evaluate and the continuous and categorical methods we would like to use.

We then fit the calculator on our `reference` data. The fitted calculator is then used to evaluate drift for the
`analysis` data, stored here as the variable ``results``.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Working with results.ipynb
    :cells: 2

This variable is an instance of the :class:`~nannyml.drift.univariate.result.Result` class. To turn this object into a
`DataFrame` you can use the :meth:`~nannyml.drift.univariate.results.Result.to_df` method. Let's see what
this `DataFrame` looks like.


.. nbimport::
    :path: ./example_notebooks/Tutorial - Working with results.ipynb
    :cells: 3

We can immediately see that the a MultiLevel index is being used to store the data. There is a part containing `chunk`
information, followed by the numerical results of the drift calculations.

In the case of the :class:`~nannyml.drift.univariate.calculator.UnivariateDriftCalculator` there are two degrees of
freedom. You can specify columns to include in the calculation, and each column might be evaluated by different methods.

This structure is visible in the column index. The top level represents the column names. The middle level represents
the specific methods used to evaluate a column. The bottom level contains the information relevant to each method:
a value, upper and lower thresholds for alerts and whether the evaluated method crossed the thresholds for that chunk,
leading to an alert.

.. nbtable::
    :path: ./example_notebooks/Tutorial - Working with results.ipynb
    :cell: 4

Working with the `Multilevel indexes` can be very powerful, yet also quite challenging.
The following snippet illustrates how to retrieve all calculated method values from our results.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Working with results.ipynb
    :cells: 5
    :show_output:

To improve this experience we've introduced a helper method that allows you to filter the result data so you can easily
retrieve the information you want. Since the :class:`~nannyml.drift.univariate.calculator.UnivariateDriftCalculator` has
two degrees of freedom we've included both in the :meth:`~nannyml.drift.univariate.result.Result.filter` method.
Additionally you can filter on the :term:`data period<Data Period>`, i.e. ``reference`` or ``analysis``.

The :meth:`~nannyml.drift.univariate.result.Result.filter` method will return a new
:class:`~nannyml.drift.univariate.result.Result` instance, allowing you to chain methods like,
:meth:`~nannyml.drift.univariate.result.Result.filter`, :meth:`~nannyml.drift.univariate.result.Result.to_df` and
:meth:`~nannyml.drift.univariate.result.Result.plot`.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Working with results.ipynb
    :cells: 6
    :show_output:

When looking at the results after filtering, you can see only the `chi2` data for the `salary_range` column during the
`analysis` period is included.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Working with results.ipynb
    :cells: 7

.. nbtable::
    :path: ./example_notebooks/Tutorial - Working with results.ipynb
    :cell: 8

To avoid the use of a `Multilevel index`, we've provided a switch in the
:meth:`~nannyml.drift.univariate.result.Result.to_df` method.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Working with results.ipynb
    :cells: 9

.. nbtable::
    :path: ./example_notebooks/Tutorial - Working with results.ipynb
    :cell: 10

Results can also be exported to external storage using a :class:`~nannyml.io.base.Writer`. We currently support writing
results to disk using a :class:`~nannyml.io.raw_files_writer.RawFilesWriter`, serializing the
:class:`~nannyml.drift.univariate.result.Result` into a Python pickle file and storing that to disk using the
:class:`~nannyml.io.pickle_file_writer.PickleFileWriter` or storing calculation results in a database using the
:class:`~nannyml.io.db.database_writer.DatabaseWriter`. This example will show how to use the
:class:`~nannyml.io.db.database_writer.DatabaseWriter`.

We construct the :class:`~nannyml.io.db.database_writer.DatabaseWriter` by providing a database connection string.
Upon calling the :meth:`~nannyml.io.db.database_writer.DatabaseWriter.write` method all results will be written into
the database, in this case a `SQLite` database.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Working with results.ipynb
    :cells: 11

A quick inspection shows the database was populated and contains the univariate drift calculation results.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Working with results.ipynb
    :cells: 12
    :show_output:

.. nbimport::
    :path: ./example_notebooks/Tutorial - Working with results.ipynb
    :cells: 13
    :show_output:
