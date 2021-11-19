"""
Submission file for Hackathon 3.
Group: <group_number>

Please, make sure this file passes tests from `validate.py` before submitting!

Do not:
- change the signature and names of functions
- import packages that where not installed during the Install session

You can:
- create any number of functions, methods, and use them inside the functions that will be assessed

Warning: plagiarism is not tolerated and we can detect it
"""

import numpy as np


def precision(cm):
    """-----------------------------------------------------------------------------------------
    Based on the confusion matrix, computes the 'precision'
    @pre:
        - cm : confusion_matrix of a binary classification
    @post:
        - score: precision (or positive predictive value), associated with cm
    -----------------------------------------------------------------------------------------"""
    # Start of the contribution
    ppv = 1

    # End of the contribution
    return ppv


def recall(cm):
    """-----------------------------------------------------------------------------------------
    Based on the confusion matrix, computes the 'recall'
    @pre:
        - cm : confusion_matrix of a binary classification
    @post:
        - r: recall (or true positive rate), associated with cm
    -----------------------------------------------------------------------------------------"""
    # Start of the contribution
    tpr = 1
    # End of the contribution
    return tpr


def probas_to_F1(y_true, y_pred, output="F1", threshold=0.5):
    """-----------------------------------------------------------------------------------------
    Evaluates the F1 score which is a harmonic mean of the precision and recall
    @pre:
        - y_true: vectors of 0 and 1 representing the real class values
        - y_pred: vectors of real values representing predicted probability of being in the class of good wines ('1')
        - output:  'F1' means that the output should only be the F1 score.
                   'PRF1' means that the output is a tuple with (precision, recall, F1)
                   'F1' is the default value
        - threshold: a threshold probability (between 0 and 1) to determine if a wine is good ('1')
    @post:
        - F1_score: harmonic mean of the precision and recall
        - If asked in argument, precision and recall can be added in the output: (precision, recall, F1)
    -----------------------------------------------------------------------------------------"""

    y_pred = pred_probas_to_pred_labels(y_pred, threshold)

    # Start of the contribution

    # TO MODIFY
    ppv = 0  # positive predictive value (or precision)
    tpr = 0  # true positive rate (or recall)
    F1_score = 0.0

    if output == "PRF1":
        return (ppv, tpr, F1_score)

    # End of the contribution

    return F1_score


def evalParam(methods, param, X, y, cv):
    """
    @pre:
        - methods: list of classifiers to analyze
        - param: list of size len(methods) containing lists of parameters (in dictionary form) to evaluate.
                 In other words, param[i][j] is a dictionary of parmeters.
                 For example if index i is for KNN, we can have a parameter configuration (with index j) described as
                     param[i][j] = {"n_neigbors":5, "weights": 'uniform'};
                     while param[i] is a list of such parameters dictionnaries for model i (here KNN)
        - X: training dataset
        - y: target vector for the corresponding entries of X
        - cv: the number of folds to use in your cross-validation
    @post:
        - score: list with same shape as param. score[i][j] = mean score over the folds,
                                                             using method i with parameters param[i][j]
    ------------------------------------------------------------------------------------------------"""
    from sklearn.model_selection import KFold

    # Start of the contribution
    score = [[0, 1], [0]]
    # TO DO
    # End of the contribution
    return score


def pred_probas_to_pred_labels(proba_vec, threshold=0.5):

    return np.where(proba_vec <= threshold, 0, 1)
