.. _instance_level_metrics:

Instance-level metric contributions
===================================

NannyML normally returns one value per data chunk. Instance-level metric
contributions are an opt-in companion output that helps identify the rows most
associated with that value. The regular chunk result, thresholds, alerts,
confidence bounds, and sampling errors are unchanged.

Interface
---------

Pass ``include_instance_metrics`` to ``calculate`` or ``estimate``. It accepts
``False`` (the default), ``True`` for every supported metric configured on the
calculator, or a list of metric names for selective calculation.

.. code-block:: python

    result = calculator.calculate(
        analysis_data,
        include_instance_metrics=['accuracy', 'roc_auc'],
        identifier_column_name='prediction_id',
    )

    chunks = result.to_df()
    instances = result.to_instance_df()

``to_df()`` retains the existing chunk-level schema. ``to_instance_df()`` is a
separate flat dataframe containing:

* the identifier column;
* ``chunk_key``, ``chunk_index``, chunk start/end indices and dates, and
  ``period``;
* one ``<metric>_contribution`` column per selected scalar result component.

The input features and predictions are deliberately not copied into this
table. Join it to the input dataframe by identifier when they are needed for
investigation. ``identifier_column_name`` must name a non-null, unique column.
When it is omitted, NannyML uses the dataframe index and calls the output
column ``identifier``. Multi-index values are represented as tuples. This also
keeps identifiers stable when timestamp-based chunking sorts the data.
Identifier names must not collide with the chunk metadata names or end in
``_contribution``.

Calling ``to_instance_df()`` after an ordinary calculation raises an
``InvalidArgumentsException`` explaining that instance metrics were not
requested. Filtering a result by period, metric, or monitored column also
filters its instance table.

Contribution direction
----------------------

A contribution has the natural direction of its metric; it is not converted
to an anomaly score. For a beneficial metric such as accuracy, a high value is
favourable. For a loss, missing-value indicator, reconstruction error, or
domain-classifier AUROC, a high value is unfavourable. Investigations should
therefore sort in the direction of the alert: ascending for a lower-bound
performance alert and descending for an upper-bound error or drift alert.

Realized performance
--------------------

Regression
~~~~~~~~~~

* **MAE:** absolute error; its chunk reducer is the mean.
* **MAPE:** absolute percentage error using scikit-learn's epsilon denominator;
  its reducer is the mean.
* **MSE:** squared error; its reducer is the mean.
* **MSLE:** squared ``log1p`` error; its reducer is the mean.
* **RMSE:** squared error; its reducer is the square root of the mean.
* **RMSLE:** squared ``log1p`` error; its reducer is the square root of the
  mean.

Binary classification
~~~~~~~~~~~~~~~~~~~~~

* **Accuracy:** 1 for a correct prediction and 0 otherwise; the mean is the
  chunk accuracy.
* **Precision:** 1 or 0 for predicted-positive rows and ``NaN`` for other
  rows; the mean ignoring ``NaN`` is precision.
* **Recall:** 1 or 0 for actual-positive rows and ``NaN`` for other rows; the
  mean ignoring ``NaN`` is recall.
* **Specificity:** 1 or 0 for actual-negative rows and ``NaN`` for other rows;
  the mean ignoring ``NaN`` is specificity.
* **F1:** 1 for true positives, 0 for false positives and false negatives, and
  ``NaN`` for true negatives. The mean ignoring ``NaN`` is the Jaccard score
  ``J``; F1 is ``2J / (1 + J)``.
* **ROC AUC:** the Mann--Whitney concordance of the row against every member of
  the opposite class, including half credit for ties. Its mean is ROC AUC.
* **Average precision:** positive baseline credit with every ranking-inversion
  penalty split equally between its positive and negative endpoints. The sum
  is average precision and poorly ranked negative rows receive negative
  contributions.
* **Confusion matrix:** one column per cell. Each row contributes its cell
  indicator, adjusted for the configured normalization; each column sums to
  the corresponding chunk cell.
* **Business value:** the row's payoff. With ``per_prediction`` normalization,
  it is divided by the number of predictions in that predicted class. The sum
  is chunk business value.

Multiclass classification
~~~~~~~~~~~~~~~~~~~~~~~~~

Accuracy uses the same 0/1 mean as binary accuracy. ROC AUC uses per-class
one-vs-rest Mann--Whitney concordance followed by the existing macro average;
the row mean is the macro ROC AUC. Average precision uses the symmetric rank
allocation per one-vs-rest class and its row contributions sum to macro average
precision. Precision, recall, specificity, and F1 allocate each class's macro
term to its participating rows, so their row contributions sum to the chunk
metric. Confusion-matrix cells and business value are allocated as in the
binary case and also sum to their chunk values.

Confidence-based performance estimation
---------------------------------------

CBPE uses calibrated probabilities as expected target membership and retains
uncalibrated model scores for ranking metrics.

* **Accuracy:** expected correctness; the mean is estimated accuracy.
* **Precision:** calibrated positive probability on predicted-positive rows
  and ``NaN`` otherwise; the mean is estimated precision. If a chunk has no
  predicted positives, every valid row contributes zero, matching the
  estimator's zero-denominator behavior.
* **Recall:** positive probability for predicted positives and negative
  positive-probability for predicted negatives. This signed value is intended
  for ranking false-negative pressure; recall is a ratio and is not a direct
  mean or sum of this column.
* **Specificity:** expected-negative probability for predicted negatives and
  its negative for predicted positives. It is a diagnostic signed contribution
  rather than a direct additive reducer.
* **F1:** positive probability for predicted positives and its negative for
  predicted negatives. It preserves the useful row ordering; F1 remains a
  non-linear ratio.
* **ROC AUC:** probabilistic Mann--Whitney concordance. A constant
  reconciliation offset accounts for the existing estimator's cumulative-count
  rounding and stable ordering of tied scores. The offset does not change the
  row ordering, and the mean is the estimated ROC AUC.
* **Average precision:** expected positive credit and symmetric expected
  inversion penalties. Contributions sum to the existing estimated average
  precision, including its cumulative-count rounding behavior.
* **Confusion matrix:** expected cell mass, adjusted for normalization. Each
  component column sums to its estimated cell.
* **Business value:** expected payoff under the calibrated class probabilities;
  contributions sum to the estimate.

For multiclass CBPE, the calculations are performed one-vs-rest and collapsed
using the existing macro average. Accuracy is a mean. Precision, recall,
specificity, F1, average precision, confusion-matrix cells, and business value
are additive and sum to their estimates. ROC AUC applies the reconciliation to
each one-vs-rest class before the macro collapse, so its row mean is the macro
estimate.

Direct loss estimation
----------------------

DLE already predicts a loss for each row. The contribution is that internal
prediction after clipping negative predictions to zero. MAE, MAPE, MSE, and
MSLE use the mean. RMSE and RMSLE use the square root of the mean. Selecting
instance metrics does not fit additional loss models.

Data quality and summary statistics
-----------------------------------

Missing values, unseen values, and out-of-range values expose 0/1 indicators.
The reducer is the mean when the calculator is normalized and the sum when it
returns counts.

Average, sum, and median calculators expose each row's raw value. Standard
deviation exposes squared deviation from the chunk mean; the chunk statistic is
``sqrt(sum(contribution) / (valid_count - 1))``. Row count exposes 1 for every
row and uses the sum.

For these single-purpose, per-column calculators, a selective list contains
monitored column names. Row count accepts ``rows_count``.

Multivariate drift
------------------

* **Data reconstruction:** the Euclidean PCA reconstruction norm already used
  internally; its mean is the chunk reconstruction error.
* **Domain classifier:** each analysis row's out-of-fold classifier score is
  converted to its Mann--Whitney concordance against all out-of-fold reference
  scores. The mean is the domain-classifier AUROC. Contributions reuse the same
  cross-validation fits as the chunk result; no additional classifier is fit.

Scope and cost
--------------

Instance metrics are disabled by default so existing performance and memory
usage remain unchanged. Select only the metrics needed for an investigation
when rank calculations or wide confusion matrices would otherwise be costly.

Univariate drift distances (Kolmogorov--Smirnov, Jensen--Shannon, Hellinger,
Wasserstein, L-infinity, and chi-square) are intentionally excluded because a
useful row influence would require expensive leave-one-out recomputation.
Continuous and categorical distribution calculators are also outside this
interface. Their existing chunk-level behavior is unchanged, and requesting
instance metrics from an unsupported calculator fails explicitly.
