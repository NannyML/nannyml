.. _chunk-data:
====
Chunking data
====


Placeholder...

Different partitions within one chunk
====
If you want to get performance estimation or data drift results for a dataset that contains two
partitions - *reference* and *analysis* (see :term:`Partition`), most likely
there will be a chunk that contains both of them. We call it transition chunk. All the chunks before belong to
*reference* period
and all after, based on *analysis* period, are *actual* results. This is especially important for Performance Estimation
(# TODO naming?), where *reference* period should be treated like you treat your train set when modelling whereas
*analysis* is like test - the quality of estimation on the *reference* will most likely be much better than on
*analysis*.

It may happen that there is no transition chunk, in that case (# TODO)
