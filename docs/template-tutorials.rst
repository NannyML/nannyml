=================
[Page title]
=================

This first paragraph or two should explain what the feature does in very simple terms. If more detailed explanations are required, include a link to a page in the "How it works" section.


.. code-block:: python

    All the code that is run in the tutorial should be presented here.
    If it is too big for a code-block (~50+ lines) then it should be 
    presented as a link to an example notebook instead.


Walkthrough
===========

[Step title]
~~~~~~~~~~~~

Any required context in brief, with links to any relevant "How It Works" articles if deeper knowledge may be required.

Succinct description of what the code does.

.. code-block:: python

    Code examples should use one of the sample datasets 
    we package with the library.


Samples of returned dataframes as tables should be presented wherever necessary:

+-------+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+---------------------+----------------+-------------+
|       |   distance_from_office | salary_range   |   gas_price_per_litre |   public_transportation_cost | wfh_prev_workday   | workday   |   tenure |   identifier | timestamp           |   y_pred_proba | partition   |
+=======+========================+================+=======================+==============================+====================+===========+==========+==============+=====================+================+=============+
| 49995 |                6.04391 | 0 - 20K €      |               1.98303 |                      5.89122 | True               | Thursday  |  6.41158 |        99995 | 2021-01-01 02:42:38 |           0.17 | analysis    |
+-------+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+---------------------+----------------+-------------+
| 49996 |                5.67666 | 20K - 20K €    |               2.04855 |                      7.5841  | True               | Wednesday |  3.86351 |        99996 | 2021-01-01 04:04:01 |           0.55 | analysis    |
+-------+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+---------------------+----------------+-------------+
| 49997 |                3.14311 | 0 - 20K €      |               2.2082  |                      6.57467 | True               | Tuesday   |  6.46297 |        99997 | 2021-01-01 04:12:57 |           0.22 | analysis    |
+-------+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+---------------------+----------------+-------------+
| 49998 |                8.33514 | 40K - 60K €    |               2.39448 |                      5.25745 | True               | Monday    |  6.40706 |        99998 | 2021-01-01 04:17:41 |           0.02 | analysis    |
+-------+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+---------------------+----------------+-------------+
| 49999 |                8.26605 | 0 - 20K €      |               1.41597 |                      8.10898 | False              | Friday    |  6.90411 |        99999 | 2021-01-01 04:29:32 |           0.02 | analysis    |
+-------+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+---------------------+----------------+-------------+

Plots should be added either as images or (if possible) interactive items.

.. image:: ./_static/perf-est-guide-syth-example.svg

Outcome
======================================

Insights
~~~~~~~~~

What could a user understand from doing this activity on different data? No more than 3 short paragraphs, with links to examples where relevant.

What next?
~~~~~~~~~~

Recommendations on what the user could do next, whether inside NannyML library or otherwise. Link to relevant things. No more than three short paragraphs.
