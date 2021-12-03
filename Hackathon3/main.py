"""
Submission file for Hackathon 3.
Group: 2

Please, make sure this file passes tests from `validate.py` before submitting!

Do not:
- change the signature and names of functions
- import packages that where not installed during the Install session

You can:
- create any number of functions, methods, and use them inside the functions that will be assessed

Warning: plagiarism is not tolerated and we can detect it
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold


def precision(cm):
    """-----------------------------------------------------------------------------------------
    Based on the confusion matrix, computes the 'precision'
    @pre:
        - cm : confusion_matrix of a binary classification
    @post:
        - score: precision (or positive predictive value), associated with cm
    -----------------------------------------------------------------------------------------"""
    _, fp, _, tp = cm.ravel()
    ppv = tp/(tp+fp)
    return ppv


def recall(cm):
    """-----------------------------------------------------------------------------------------
    Based on the confusion matrix, computes the 'recall'
    @pre:
        - cm : confusion_matrix of a binary classification
    @post:
        - r: recall (or true positive rate), associated with cm
    -----------------------------------------------------------------------------------------"""
    _, _, fn, tp = cm.ravel()
    tpr = tp/(tp+fn)
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

    cm = confusion_matrix(y_true, y_pred)
    ppv = precision(cm)
    tpr = recall(cm)
    F1_score = 2*ppv*tpr/(ppv+tpr) 

    return (ppv, tpr, F1_score) if output == "PRF1" else F1_score


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
    score = [np.zeros(len(param[i])) for i in range(len(methods))]
    X = np.array(X)
    y = np.array(y)
    
    kf = KFold(n_splits=cv)

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index, :], X[val_index, :]
        y_train, y_val = y[train_index].ravel(), y[val_index].ravel()

        for m in range(len(methods)):
            for p in range(len(param[m])):
                methods[m].set_params(**param[m][p])
                methods[m].fit(X_train, y_train)
                y_pred = methods[m].predict(X_val)
                score[m][p] += probas_to_F1(y_val, y_pred)/cv

    return score


def pred_probas_to_pred_labels(proba_vec, threshold=0.5):
    return np.where(proba_vec <= threshold, 0, 1)
