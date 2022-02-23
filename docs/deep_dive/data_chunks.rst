====================
Data Chunks
====================

Not sure what data chunk is in the first place? Go here (# TODO link to glossary).

Why we need chunks?
====
NannyML monitors your model performance and input data changes. Both can be reliably evaluated only on samples
of data containing a number of observations. We call these samples chunks. All the results that we provide are
calculated and presented on the level of chunk i.e. a chunk is a single data point. (# TODO example plot here)


How you define chunk?
====
Examples here from data drift on size-based and time-based chunks.

Minimum chunk size
======
**In data sciences sample size affects everything, especially when it is small**. NannyML lets you decide on the way
you split your data in chunks. That is because periods in the data may be meaningful and no one knows it better than
you.
However, when the chunks are too small, **what looks like a severe drop in performance of your model, may in fact be
only sampling effect**. To better understand that, look at the histogram below. It shows dispersion of ROC AUC for
random model *predicting* random target (which by definition should be 0.5).

When there are many chunks, it is easy to spot the noisy nature of fluctuations. However, if you have only few chunks
in the *analysis* period, you may get confused. In order to minimize this risk we estimate a minimum chunk size for
your data.
Since NannyML is performance-oriented, the minimum chunk size is estimated in order to keep variation of performance
of your model low. Our definition of *low* is arbitrary for now:
 - For models with ROC AUC below 0.9 we want to have chunks for which standard deviation is lower
   than 0.01.
 - For other models, standard deviation of performance on chunks should be below 0.02.

For the sake of computation time we did not want to iterate on your data to get the chunk size that meets the
requirement on dispersion level. We know, that sample variation of ROC AUC is mostly affected by the quality of the model and the
class balance. We have
ran experiments on synthetic data to quantify that. For each artificially created vector of ground truths and
predicted probabilities we draw sample of constant size many times and measured the dispersion. We did
that for different sample sizes. Then we choose only the experiments that fulfill our requirements on standard
deviation value and fitted linear regression (after transforming to 3rd order polynomial). As a result we have a
function of two arguments - ROC AUC score and class balance calculated on your
*reference* data that returns suggested minimum chunk size. The output of that function is limited
with hard floor of 500 observations per sample. See the plots of experiment results and fitted surface.

.. image:: ../_static/deep_dive_data_chunks_minimum_chunk_size.svg
    :width: 800pt

It is easy to imagine two different datasets and models with ROC AUC scores and class balances that are the same,
but dispersions of ROC AUC on samples of the same size that are different. Moreover, the arbitrary limits on standard
deviation may not fit all the cases. After all, there are situations where the performance actually fluctuates on
*reference* data (due to e.g. seasonality). Finally, there are cases where only one chunk size makes sens (e.g.
weekly split). For this reasons, **minimum chunk size should be never treated neither as recommended chunk size nor
as a hard limit**. It is just a chunk size, below which performance - actual or estimated - most likely will be
governed by sampling rather than actual changes. Finally, be aware that sample size affects also calculations related
to data drift.

Different partitions within one chunk
====
If you want to get performance estimation or data drift results for a dataset that contains two
partitions - *reference*
and *analysis* (# TODO link to
glossary), most likely
there will be a chunk that contains both of them. We call it transition chunk. All the chunks before belong to
*reference* period
and all after, based on *analysis* period, are *actual* results. This is especially important for Performance Estimation
(# TODO naming?), where *reference* period should be treated like you treat your train set when modelling whereas
*analysis* is like test - the quality of estimation on the *reference* will most likely be much better than on
*analysis*.

It may happen that there is no transition chunk, in that case (# TODO)

