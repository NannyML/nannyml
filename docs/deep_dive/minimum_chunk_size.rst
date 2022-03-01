.. _minimum-chunk-size:

==================
Minimum chunk size
==================

.. note::
    Not sure what data chunk is in the first place? Read about :term:`Data Chunk` for short explanation or go through
    the :ref:`guide<chunk-data>` for practical implementations.


**In data sciences sample size affects everything, especially when it is small**. NannyML allows to split data
in chunks in different ways because periods in the data may be meaningful and no one knows it better than
the owner of the monitored model/data.
However, when the chunks are too small, **what looks like a significant drop in performance of the monitored model,
may in
fact be only sampling effect**. To better understand that, look at the histogram below. It
shows
dispersion of ROC AUC for random model *predicting* random binary target (which by definition should be 0.5) for sample
of 100 observations. It is not uncommon to get ROC AUC of 0.65 for some samples.

.. code-block:: python

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.metrics import roc_auc_score

    >>> sample_size = 100
    >>> roc_aucs = []

    >>> for experiment in range(10_000):
    >>>     y_trues = np.random.binomial(1, 0.5, sample_size) # balanced dataset
    >>>     y_pred_probas = np.random.beta(0.5,0.5, sample_size) # beta distribution of y_pred_proba
    >>>     roc_aucs.append(roc_auc_score(y_trues, y_pred_probas))

    >>> plt.hist(roc_aucs, bins=50, density=True)
    >>> plt.title("ROC AUC of random classifier\n for randomly selected samples of 100 observations.");

.. image:: ../_static/deep_dive_data_chunks_stability_of_ROC_AUC.svg
    :width: 400pt

When there are many chunks, it is easy to spot the noisy nature of fluctuations. However, with only few chunks it
is difficult to tell whether the effect (the drop) is real. To minimize this risk, NannyML estimates a minimum chunk
size for the monitored data and raises a warning if the selected split results in chunks that are smaller.
Since NannyML is performance-oriented, the minimum chunk size is estimated in order to keep variation of performance
of your model *low*. *Low* is defined as:

- For models with ROC AUC below 0.9, standard deviation of ROC AUC on chunks should be lower than 0.01.
- For other models, standard deviation of ROC AUC on chunks should be below 0.02.

Typical way to approach the task of finding minimum chunk size would be to iterate on the monitored data to find the
smallest chunk size that meets the above requirements. This in some cases could be resource intensive, so instead
NannyML uses simple model to quickly estimate that based on characteristics of the monitored data.
Experiments have shown that variability of ROC AUC with respect to sample size is mostly affected
by the quality of the monitored model (i.e. its performance) and the target distribution (class balance). In order to
quantify the impact, a large
number of synthetic data sets was created with different target distributions and models of different quality. For each
artificially created vector of ground truths and predicted probabilities a sample of constant size was drawn many times
and the standard deviation was calculated. Then, the experiments that met the requirements on standard
deviation value were chosen and a model was fitted (see experiment results and fitted surface on the plot below).
As a result, a function of two arguments - ROC AUC score and target distribution - was obtained.
NannyML uses this function to calculate minimum chunk size based on the characteristics of monitored data.
If any of the created chunks is smaller than the minimum estimated, a warning
is raised.

.. image:: ../_static/deep_dive_data_chunks_minimum_chunk_size.svg
    :width: 800pt

It is easy to imagine two different datasets and models with ROC AUC scores and class balances that are the same,
but dispersions of ROC AUC on samples of the same size that are different. Moreover, the arbitrary limits on standard
deviation may not fit all the cases. After all, there are situations where the performance actually fluctuates on
*reference* data (due to e.g. seasonality). Finally, there are cases where only one chunk size is justified from
business perspective (e.g. quarterly split). For this reasons, **minimum chunk size should be never treated neither as
recommended chunk size nor
as a hard limit**. It is just a chunk size, below which performance - actual or estimated - most likely will be
governed by sampling rather than actual changes. Finally, be aware that sample size affects also calculations related
to data drift.
