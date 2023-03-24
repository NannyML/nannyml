.. _why-estimate-performance:

Why Estimate Performance
============================================

NannyML allows estimating the performance of ML models when :term:`targets<Target>` are absent.
This can be very helpful when targets are delayed, only partially available, or not available at all, because
it allows you to potentially identify performance issues before they would otherwise be detected.

Some specific examples of when you could benefit from estimating your performance include:

- When predicting loan defaults, to estimate model performance before the end of the repayment periods.
- When performing sentiment analysis, targets may be entirely unavailable without significant human effort, so estimation is the only feasible way to attain metrics.
- When dealing with huge datasets, where human verification can only cover a small sample, estimation of performance can help confirm confidence or question the efficacy.

We offer performance estimation for *binary classification*, *multiclass classification*, and *regression* tasks.
To see how to use NannyML's performance estimation, check out the following tutorials:

- :ref:`binary-performance-estimation`
- :ref:`multiclass-performance-estimation`
- :ref:`regression-performance-estimation`

To learn about the underlying principles and algorithms we use for performance estimation, refer to these sections:

- :ref:`How Performance Estimation Works for Classification Tasks<how-it-works-cbpe>`
- :ref:`How Performance Estimation Works for Regression Tasks<how-it-works-dle>`
