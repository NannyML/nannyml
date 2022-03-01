.. _performance-estimation-deep-dive:

=======================================
Confidence-based Performance Estimation
=======================================

Introduction
============

A binary classification model typically returns two outputs for each prediction - a class (binary) and a class
probability predictions (sometimes referred to as score). The score provides information about the confidence of the
prediction. A rule of thumb is that, the closer a score is to its lower or upper limit (usually 0 and 1), the higher
the probability that a classifier’s prediction is correct. When this score is an actual probability, it can be
directly used to calculate the probability of making an error. For instance, imagine a high-performing model which,
for a large set of observations, returns a prediction of 1 (positive class) with probability of 0.9. It means that
for approximately 90% of these observations, the model is correct while for the other 10% the model is wrong.
Assuming properly calibrated probabilities, NannyML reconstructs the whole confusion matrix as follows:

    1. Threshold all the probability outputs to get binary predictions for all the observations. If the probability
       assigned to a single observation is greater or equal to the threshold, it is a **positive** prediction. If
       it is lower - it is a **negative** prediction.
    2. The probability that the prediction is false - **P(F) = abs(predicted class - predicted probability)**.
       Then **P(T)= 1 - P(F)**.
    3. The probability of True Positive, False Positive, True Negative and False negative is calculated.

For example, for probability equal to 0.9 and threshold being 0.5, the prediction is:

    - positive as 0.9 > 0.5,
    - false (False Positive) with 0.1 (1-0.9) probability,
    - true (True Positive) with 0.9 probability,
    - True Negative and False Negative with 0 probability.

Summing these values for all observations in a chunk gives the expected confusion matrix for a selected threshold. This
can be done for all the thresholds (all the model output probabilities) and then used to calculate the estimated ROC
AUC. Using simple modifications, the future versions of NannyML will estimate any metric for any supervised learning problem.

Since the probability returned by the model contains information about its confidence in prediction, this family of methods is called Confidence-based Performance Estimation (CBPE).

Probability calibration
=======================
In order to accurately estimate the performance from the model scores, they need to be well calibrated. If a classifier assigns a probability of 0.9 for a set of observations and 90% of these observations belong to the positive class, we consider that classifier to be well calibrated with respect to that subset. Most predictive models focus on performance rather than on probability estimation, therefore their scores are rarely calibrated.
Examples of different models and their calibration curves are shown below [1]_:

.. image:: ../_static/deep_dive_performance_estimation_calibration_curves.png

Probabilities can be calibrated in post-processing. NannyML uses isotonic regression to
calibrate model scores [1]_ [2]_. Since some of the models
are probabilistic and their probabilities are calibrated by design, first NannyML checks whether calibration is required
. It is done according to the following logic:

1. Stratified shuffle split [3]_ (controlled for the positive class) of reference data into 3 folds.
2. For each fold, a calibrator is fitted on train and *predicts* new probabilities for test.
3. Quality of calibration is evaluated by comparing the Expected Calibration Error (ECE) [4]_ for the raw and calibrated
   (predicted) probabilities on the test splits:


    - If in any of the folds the ECE score is higher after post processing (i.e. calibration curve is worse), the
      calibration will not be performed.

    - If in each fold post processing improves the quality of calibration, the calibrator is fitted on the whole
      reference set and probabilities are calibrated on the set that is subject to analysis.

Calibrating probabilities is yet another reason why NannyML requires reference data that is not a training set of the monitored model.
Fitting a calibrator on model training data would introduce bias [1]_.

**References**

.. [1] https://scikit-learn.org/stable/modules/calibration.html
.. [2] https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html
.. [3] https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
.. [4] Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht: "Obtaining well calibrated probabilities using bayesian binning." Twenty-Ninth AAAI Conference on Artificial Intelligence, 2015.
