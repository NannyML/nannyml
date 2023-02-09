.. _chunking:

======================================
Chunking
======================================

Why do we need chunks?
----------------------

NannyML monitors ML models in production by doing data drift detection and performance estimation or monitoring.
This functionality relies on aggregate metrics that are evaluated on samples of production data.
These samples are called :term:`chunks<Data Chunk>`.

All the results generated are
calculated and presented per chunk i.e. a chunk is a single data point on the monitoring results. You
can refer to :ref:`Data Drift guide<data-drift>` or :ref:`Performance Estimation guide<performance-estimation>`
to review example results.



Walkthrough on creating chunks
------------------------------

To allow for flexibility there are many ways to create chunks. The examples below will show how different
kinds of chunks can be created. The examples will be run based on the performance estimation flow on the
synthetic binary classification dataset provided by NannyML. First we set up this dataset.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Chunking - Car Loan.ipynb
    :cells: 1

Time-based chunking
~~~~~~~~~~~~~~~~~~~
soo

Time-based chunking creates chunks based on time intervals. One chunk can contain all the observations
from a single hour, day, week, month etc. In most cases, such chunks will vary in the number of observations they
contain. Specify the ``chunk_period`` argument to get appropriate split. The example below chunks data quarterly.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Chunking - Car Loan.ipynb
    :cells: 2

.. nbtable::
    :path: ./example_notebooks/Tutorial - Chunking - Car Loan.ipynb
    :cell: 3

.. note::
    Notice that each calendar quarter was taken into account, even if it was not fully covered with records.
    This means some chunks contain fewer observations (usually the last and the first). See the first row above - Q3 is
    July-September, but the first record in the data is from the last day of August. The first chunk has ~1200 of
    observations while the 2nd and 3rd contain above 3000. This can cause some chunks to be less reliably estimated or calculated.

Possible time offsets are listed in the table below:

+------------+------------+
| Alias      | Description|
+============+============+
| S          | second     |
+------------+------------+
| T, min     | minute     |
+------------+------------+
| H          | hour       |
+------------+------------+
| D          | day        |
+------------+------------+
| W          | week       |
+------------+------------+
| M          | month      |
+------------+------------+
| Q          | quarter    |
+------------+------------+
| A, y       | year       |
+------------+------------+


Size-based chunking
~~~~~~~~~~~~~~~~~~~

Chunks can be of fixed size, i.e. each chunk contains the same number of observations. Set this up by specifying the
``chunk_size`` parameter.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Chunking - Car Loan.ipynb
    :cells: 4

.. nbtable::
    :path: ./example_notebooks/Tutorial - Chunking - Car Loan.ipynb
    :cell: 5

.. note::
    If the number of observations is not divisible by the chunk size required, the number of rows equal to the
    remainder of a division will be dropped. This ensures that each chunk has the same size, but in worst case
    scenario it results in dropping ``chunk_size-1`` rows. Notice that the last index in the last chunk is 48999
    while the last index in the raw data is 49999:

    .. nbimport::
        :path: ./example_notebooks/Tutorial - Chunking - Car Loan.ipynb
        :cells: 6

    .. nbtable::
        :path: ./example_notebooks/Tutorial - Chunking - Car Loan.ipynb
        :cell: 7

    .. nbimport::
        :path: ./example_notebooks/Tutorial - Chunking - Car Loan.ipynb
        :cells: 8
        :show_output:


Number-based chunking
~~~~~~~~~~~~~~~~~~~~~

The total number of chunks can be set by the ``chunk_number`` parameter:

.. nbimport::
    :path: ./example_notebooks/Tutorial - Chunking - Car Loan.ipynb
    :cells: 9
    :show_output:

.. note::
    Chunks created this way will be equal in size. If the number of observations is not divisible by the
    ``chunk_number`` then the number of observations equal to the residual of the division will be dropped.

    .. nbimport::
        :path: ./example_notebooks/Tutorial - Chunking - Car Loan.ipynb
        :cells: 10

    .. nbtable::
        :path: ./example_notebooks/Tutorial - Chunking - Car Loan.ipynb
        :cell: 11

    .. nbimport::
        :path: ./example_notebooks/Tutorial - Chunking - Car Loan.ipynb
        :cells: 12
        :show_output:

.. note::
    The same splitting rule is always applied to the dataset used for fitting (``reference``) and the dataset of
    interest (in the presented case - ``analysis``). Unless these two datasets are of the same size, the chunk sizes
    can be considerably different. E.g. if the ``reference`` dataset has 10 000 observations and the ``analysis``
    dataset has 80 000, and chunking is number-based, chunks in ``reference`` will be much smaller than in
    ``analysis``. Additionally, if the data drift or performance estimation is calculated on
    combined ``reference`` and ``analysis`` the results presented for ``reference`` will be calculated on different
    chunks than they were fitted.

Automatic chunking
~~~~~~~~~~~~~~~~~~

The default chunking method is count-based, with the desired count set to `10`. This is used if a chunking method isn't specified.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Chunking - Car Loan.ipynb
    :cells: 13
    :show_output:


Chunks on plots with results
----------------------------

Finally, once the chunking method is selected, the full performance estimation can be run.

Each point on the plot represents a single chunk, with the y-axis showing the performance.
They are aligned on the x axis with the date at the end of the chunk, not the date in the middle of the chunk.
Plots are interactive - hovering over the point will display the precise information about the period,
to help prevent any confusion.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Chunking - Car Loan.ipynb
    :cells: 14

.. image:: /_static/tutorials/chunking/chunk-size.svg
