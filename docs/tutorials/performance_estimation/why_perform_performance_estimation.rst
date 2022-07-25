.. _why-perform-performance-estimation:

Why Perform Performance Estimation
============================================

NannyML allows estimating the performance of ML models when :term:`targets<Target>` are absent.
This can be very helpful when targets are delayed, only partially available, or not available at all, because
it allows you to potentially identify performance issues before they would otherwise be detected.

Some specific examples of when you could benefit from estimating your performance include:

- When predicting loan defaults, to estimate model performance before the end of the repayment periods.
- When performing sentiment analysis, targets may be entirely unavailable without significant human effort, so estimation is the only feasible way to attain metrics.
- When dealing with huge datasets, where human verification can only cover a small sample, estimation of performance can help confirm confidence or question the efficacy.

The following tutorials explain how to use NannyML to estimate the performance of ML
models in the absence of target data. To find out how CBPE estimates performance, read the :ref:`explanation of Confidence-based
Performance Estimation<performance-estimation-deep-dive>`.
