"""
Simply run this as `python validate.py` to make sure that your `main.py` file passes basic tests.

This, however, does not ensure that your code is correct: further tests will be run after submission for that purpose.
"""
from functools import wraps
import traceback
import textwrap

import pandas as pd
import numpy as np

filename = "Data/netflix_titles.csv"

df = pd.read_csv(filename)

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


@validate
def validate_metric():
    from main import distance_metric
    from main import generate_metric_input_vector
    from main import generate_metric_input_kwargs

    a = generate_metric_input_vector()
    b = generate_metric_input_vector()
    kwargs = generate_metric_input_kwargs()

    r = distance_metric(a, b, **kwargs)

    assert np.isscalar(r), "Metric should return a scalar value"


@validate
def validate_KPrototypes():
    from main import KPrototypes
    from main import generate_KPrototypes_input_matrix
    from main import distance_metric
    from main import generate_metric_input_kwargs

    X = generate_KPrototypes_input_matrix()
    kwargs = generate_metric_input_kwargs()

    k = 5


    cluster_labels, centroids = KPrototypes( X, k=5, n_max=1, metric=distance_metric, **kwargs)

    assert cluster_labels.shape[0] == X.shape[0], "Wrong output shape for clusters"
    assert centroids.shape == (k, X.shape[1]), "Wrong output shape for centroids"


if __name__ == "__main__":
    for func in funcs:
        func()
