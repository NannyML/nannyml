.. _performance-estimation-deep-dive:

======================
Performance Estimation
======================

Confidence-based Performance Estimation
===========================================

A binary classification model for each predictions returns two outputs - class (binary) and class probability
estimate (sometimes referred to as *score*).
Score provides information on
confidence of the prediction. The qualitative rule is that closer the score is to its lower or upper limit (usually 0
or 1), the higher the probability that a classifier is correct with its prediction. When this score is an actual
probability, it can be directly used to calculate the probability of making an error. For instance, imagine a
high-performing model which for large set of observations returned prediction of 1 (positive class) with probability
of 0.9. It means that for approximately 90% of these observations model is correct (true
positives) while for other 10% is wrong (false positives). This simple approach allows to calculate any metric for
binary classifier. Specifically, when estimating ROC AUC score, whole confusion matrix (TP, FP, TN, FN)
for each threshold is calculated.
Since the probability returned by the model contains information about its
confidence in prediction, we call this family of methods: Confidence-based Performance Estimation (CBPE).


Probability calibration
=======================
In order to accurately estimate performance from scores, these need to be well calibrated probabilities. Output
probabilities
are well calibrated when the fraction of positive observations among the observations to which a binary classifier
assigned
specific value of probability is approximately equal to that probability. Predictive models focus rather on
performance than on probability estimation, therefore scores of most of them are not calibrated. Examples of different
models
and their calibration curves are shown below [1]_:

.. image:: ../_static/deep_dive_performance_estimation_calibration_curves.png

The good information is that scores can be calibrated in post processing. NannyML uses isotonic regression to
calibrate model scores [1]_ [2]_. Since some of the models
are probabilistic and their probabilities are calibrated by design, first NannyML checks whether calibration is required
. It is done according to the following logic:

1. Stratified shuffle split [3]_ (controlled for positive class) of reference data is performed for 3 folds.
2. For each fold, a calibrator is fitted on train and *predicts* new probabilities for test.
3. Quality of calibration is evaluated by comparing Expected Calibration Error (ECE) [4]_ for raw and calibrated
   (predicted) probabilities on test:

    - If in any of the folds ECE score is higher after calibration (i.e. our calibration makes things worse)
      calibration will not be performed.

    - If in each fold post processing improves the quality of calibration, calibrator is fitted on whole reference set
      and probabilities are calibrated on the set that is subject to analysis.

Calibrating probabilities is yet another reason why NannyML requires reference data which is different then
the training data of the monitored model. Fitting calibrator on model training data would introduce bias [1]_.

**References**

.. [1] https://scikit-learn.org/stable/modules/calibration.html
.. [2] https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html
.. [3] https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
.. [4] Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht: "Obtaining well calibrated probabilities using bayesian binning." Twenty-Ninth AAAI Conference on Artificial Intelligence, 2015.
