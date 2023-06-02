#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  #
#  License: Apache Software License 2.0

"""This package implements writing Results to a database.

The result objects are converted into a more time-series like format using a `Mapper`.
Every calculator and estimator has a corresponding table where the results will be written.
Each row of these tables corresponds to a chunk in the `Result` and contains the metric value for that chunk,
threshold values etc.
"""


from .database_writer import DatabaseWriter
