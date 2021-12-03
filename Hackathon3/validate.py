"""
Simply run this as `python validate.py` to make sure that your `main.py` file passes basic tests.

This, however, does not ensure that your code is correct: further tests will be run after submission for that purpose.
"""
import numpy as np

from functools import wraps
import traceback
import textwrap


funcs = []


def validate(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            print("Testing", func, end=": ")
            func(*args, **kwargs)
            print("PASSED")
        except Exception as e:
            print("FAILED")
            print(textwrap.indent(traceback.format_exc(), "\t-> "))

    funcs.append(wrapper)

    return wrapper


"""
TOOLBOX
"""


def dimensionCheck(methods, param, score):

    rowNumber = range(len(methods))

    return all(len(param[i]) == len(score[i]) for i in rowNumber)


@validate
def validate_precision():

    from main import precision

    cm = np.array([[1, 1], [1, 0]])  # Fictional example

    ppv = precision(cm)

    assert np.isscalar(ppv), "Metric should return a scalar value"
    assert 0 <= ppv <= 1, "Metric is between 0 and 1"

    return ppv


@validate
def validate_recall():

    from main import recall

    cm = np.array([[1, 1], [1, 0]])  # Fictional example

    tpr = recall(cm)

    assert np.isscalar(tpr), "Metric should return a scalar value"
    assert 0 <= tpr <= 1, "Metric is between 0 and 1"

    return tpr


@validate
def validate_probas_to_F1():
    from main import probas_to_F1

    y_true = np.array([1, 0, 0, 1])  # Fictional example
    y_pred = np.array([1, 0.5, 0, 0.22])  # Fictional example

    F1_score = probas_to_F1(y_true, y_pred)

    assert np.isscalar(F1_score), "Metric should return a scalar value"
    assert 0 <= F1_score <= 1, "Metric is between 0 and 1"

    return F1_score


@validate
def validate_evalParam():
    from main import evalParam
    from sklearn.linear_model import LinearRegression

    linregA = LinearRegression()  # Linear Regression
    linregB = LinearRegression()  # Linear Regression

    methods = [linregA, linregB]
    paramA = [{"normalize": False}, {"normalize": True}]
    paramB = [{"normalize": False}]
    param = [paramA, paramB]

    X = np.random.rand(100, 15)
    y = (np.random.rand(100) > .5)*1
    cv = 10
    score = evalParam(methods, param, X, y, cv)
    assert isinstance(score, list), "Score is a list of arrays with variable size"
    assert len(methods) == len(
        score
    ), "Score contains n arrays, n being the number of methods "
    assert dimensionCheck(
        methods, param, score
    ), "score[i] must be an array of size equal to the ith set of parameters "

    return score


if __name__ == "__main__":
    for func in funcs:
        func()
